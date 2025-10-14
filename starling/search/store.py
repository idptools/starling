"""
SQLite Sequence Store
=====================

High-performance SQLite-backed storage for sequence metadata with compression support.

Overview
--------
The SequenceStore provides fast, indexed access to sequence data:

* **Indexed Lookups**: O(log N) access by GID, length, or hash
* **Compression**: Optional zstd compression
* **Batch Operations**: Efficient bulk insert and multi-get operations
* **Thread-Safe Reads**: Immutable read-only connections never lock
* **Atomic Writes**: Build in temp file, atomically publish when complete

Database Schema
---------------

The store uses a single table with indexes::

    CREATE TABLE sequences (
        gid       INTEGER PRIMARY KEY,  -- Global sequence ID
        len       INTEGER NOT NULL,     -- Sequence length (amino acids)
        hash8     INTEGER,               -- 8-byte SHA1 hash for dedup
        seq       BLOB NOT NULL,         -- Compressed sequence
        shard     INTEGER,               -- Source shard ID
        local_idx INTEGER,               -- Index within shard
        header    BLOB                   -- Compressed header (optional)
    );

    CREATE INDEX idx_len ON sequences(len);      -- Length-based filtering
    CREATE INDEX idx_hash8 ON sequences(hash8);  -- Hash-based dedup
    CREATE INDEX idx_header ON sequences(header);

Basic Usage
-----------

**Writing (Build Time):**

>>> from starling.search import SequenceStore
>>>
>>> # Open writer (builds in temp file)
>>> store = SequenceStore.open_writer("sequences.sqlite")
>>>
>>> # Prepare rows: (gid, len, hash8, seq_blob, shard, local_idx, header_blob)
>>> rows = []
>>> for gid, seq in enumerate(sequences):
...     seq_blob = SequenceStore.encode_seq(seq, compress=True)
...     header_blob = SequenceStore.encode_header(header, compress=True)
...     hash8 = SequenceStore.hash8(seq)
...     rows.append((gid, len(seq), hash8, seq_blob, shard_id, local_idx, header_blob))
>>>
>>> # Bulk insert (very fast)
>>> store.insert_rows(rows)
>>>
>>> # Commit and atomically publish
>>> store.close_publish()

**Reading (Query Time):**

>>> # Open reader (immutable, never locks)
>>> store = SequenceStore.open_reader("sequences.sqlite")
>>>
>>> # Get single sequence
>>> seq = store.get_seq(gid=12345)
>>>
>>> # Get header and length
>>> header, length = store.get_header_len(gid=12345)
>>>
>>> # Batch fetch (efficient)
>>> gids = [100, 200, 300, 400, 500]
>>> metadata = store.get_many_meta(gids)
>>> for gid, header, length, hash8 in metadata:
...     print(f"{gid}: {header} (len={length})")
>>>
>>> # Length-based filtering (uses index)
>>> gids_in_range = store.get_gids_by_length_range(min_len=50, max_len=500)

Compression
-----------

The store supports optional zstd compression if available.


Blob Encoding Format
--------------------

Sequences and headers are stored as BLOBs with a 1-byte flag::

    [flag: 1 byte][payload: N bytes]

    flag = 0x00: Plain UTF-8
    flag = 0x01: zstd compressed UTF-8

This allows mixing compressed and uncompressed data in the same database.


Writer Workflow
---------------

The writer uses a safe build-and-publish pattern:

1. **Build in temp file**: Unique temp path prevents conflicts
2. **Write-optimized PRAGMAs**: journal_mode=OFF, synchronous=OFF
3. **Bulk inserts**: Use executemany() for batching
4. **Atomic publish**: os.replace() ensures all-or-nothing

This prevents corrupting existing databases during builds.

Reader Workflow
---------------

Readers use immutable connections:

1. **mode=ro**: Read-only access
2. **immutable=1**: Never checks for schema changes
3. **cache=private**: Per-connection page cache
4. **Never blocks**: Multiple concurrent readers

Methods Reference
-----------------

**Class Methods:**

* ``open_writer(path)``: Create writer for building database
* ``open_reader(path)``: Open immutable reader connection

**Writer Methods:**

* ``insert_rows(rows)``: Bulk insert rows
* ``close_publish()``: Commit, optimize, and atomically publish
* ``close()``: Close connection without publishing

**Reader Methods:**

* ``get_seq(gid)``: Get sequence string by GID
* ``get_header_len(gid)``: Get (header, length) tuple
* ``get_many_header_len(gids)``: Batch fetch header+length
* ``get_many_meta(gids)``: Batch fetch header+length+hash8
* ``get_gids_by_length_range(min_len, max_len)``: Find GIDs by length

**Static Methods:**

* ``hash8(seq)``: Compute 8-byte hash of sequence
* ``encode_seq(seq, compress)``: Encode sequence to BLOB
* ``decode_seq(blob)``: Decode BLOB to sequence
* ``encode_header(header, compress)``: Encode header to BLOB
* ``decode_header(blob)``: Decode BLOB to header

Threading and Concurrency
--------------------------

**Readers:** Thread-safe and lock-free

* Multiple threads can share one reader
* Multiple processes can open separate readers
* Never blocks other readers or writers

**Writers:** Single-threaded

* One writer at a time per database
* Builds in isolated temp file
* Atomically publishes when complete

Common Patterns
---------------

**Pattern 1: Build during index creation**

>>> store = SequenceStore.open_writer("index.faiss.seqs.sqlite")
>>> for shard_id, sequences in enumerate(shards):
...     rows = [(gid, len(s), hash8(s), encode_seq(s, True), ...)
...             for gid, s in sequences]
...     store.insert_rows(rows)
>>> store.close_publish()

**Pattern 2: Query sequences during search**

>>> store = SequenceStore.open_reader("index.faiss.seqs.sqlite")
>>> gids = [result[1] for result in search_results]  # Extract GIDs
>>> metadata = store.get_many_meta(gids)
>>> for gid, header, length, hash8 in metadata:
...     seq = store.get_seq(gid)  # Lazy fetch if needed

**Pattern 3: Length-based pre-filtering**

>>> # Find all sequences of exact length (exact match search)
>>> gids = store.get_gids_by_length_range(min_len=68, max_len=68)
>>> # Use as selector for FAISS search (huge speedup!)

See Also
--------
* :class:`IndexBuilder`: Uses SequenceStore during build
* :class:`SearchEngine`: Uses SequenceStore for filtering/metadata
* :mod:`starling.search.cli`: Command-line interface

Notes
-----
* Batch operations are orders of magnitude faster than single gets
* Length-based filtering is extremely fast due to indexing
* Writer temp files are automatically cleaned up on publish
"""

from __future__ import annotations

import functools
import hashlib
import os
import sqlite3
import time
from typing import Iterable, List, Optional, Sequence, Tuple


class SequenceStore:
    """
    SQLite-backed per-gid sequence metadata store.

    Table:
      sequences(
        gid       INTEGER PRIMARY KEY,
        len       INTEGER NOT NULL,
        hash8     INTEGER,
        seq       BLOB NOT NULL,   -- 1 byte flag + payload (0=plain UTF-8, 1=zstd)
        shard     INTEGER,
        local_idx INTEGER,
        header    BLOB              -- 1 byte flag + payload (0=plain UTF-8, 1=zstd, NULL if missing)
      )
    """

    # ---------- Constructors ----------
    @classmethod
    def open_writer(cls, live_db_path: str) -> "SequenceStore":
        """
        Create a writer that builds into a UNIQUE tmp file using an IMMEDIATE
        transaction and write-optimized PRAGMAs, then later publishes atomically
        via close_publish().
        """
        pid = os.getpid()
        ts = int(time.time() * 1000)
        tmp_path_base = f"{live_db_path}.building.{pid}.{ts}"

        # We try with unix-excl first (POSIX). If not available, fallback.
        def _connect(tmp_path: str) -> sqlite3.Connection:
            # isolation_level=None => autocommit; we control txn explicitly
            # vfs=unix-excl ensures only one process can open the file at a time (POSIX).
            uri = f"file:{tmp_path}?mode=rwc&cache=private&vfs=unix-excl"
            try:
                return sqlite3.connect(
                    uri,
                    uri=True,
                    isolation_level=None,
                    check_same_thread=False,
                    timeout=0.0,  # fail fast, we retry with a fresh tmp path
                )
            except sqlite3.OperationalError:
                # Retry without vfs=unix-excl (e.g., non-POSIX or unavailable VFS)
                uri2 = f"file:{tmp_path}?mode=rwc&cache=private"
                return sqlite3.connect(
                    uri2,
                    uri=True,
                    isolation_level=None,
                    check_same_thread=False,
                    timeout=0.0,
                )

        # We may retry with a fresh unique path if BEGIN IMMEDIATE fails.
        attempt = 0
        while True:
            attempt += 1
            tmp_path = (
                tmp_path_base if attempt == 1 else f"{tmp_path_base}.retry{attempt}"
            )
            # ensure parent dir exists; ignore stale tmp cleanup (unique anyway)
            os.makedirs(os.path.dirname(live_db_path) or ".", exist_ok=True)

            conn = _connect(tmp_path)
            cur = conn.cursor()
            try:
                # Writer PRAGMAs (single-process bulk load; no WAL)
                cur.execute("PRAGMA locking_mode=EXCLUSIVE;")  # best-effort
                cur.execute("PRAGMA journal_mode=OFF;")
                cur.execute("PRAGMA synchronous=OFF;")
                cur.execute("PRAGMA temp_store=MEMORY;")
                cur.execute("PRAGMA mmap_size=268435456;")
                cur.execute("PRAGMA page_size=4096;")

                # Acquire a RESERVED write lock (blocks other writers, allows temp metadata ops)
                # IMMEDIATE is friendlier than EXCLUSIVE across filesystems.
                cur.execute("BEGIN IMMEDIATE;")

                # Create schema inside the same txn
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sequences (
                      gid       INTEGER PRIMARY KEY,
                      len       INTEGER NOT NULL,
                      hash8     INTEGER,
                      seq       BLOB NOT NULL,
                      shard     INTEGER,
                      local_idx INTEGER,
                      header    BLOB
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_len    ON sequences(len)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_hash8  ON sequences(hash8)")
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_header ON sequences(header)"
                )

                # Precompile insert statement
                insert_sql = (
                    "INSERT INTO sequences(gid,len,hash8,seq,shard,local_idx,header) "
                    "VALUES(?,?,?,?,?,?,?)"
                )
                insert_stmt = conn.cursor()
                insert_stmt.execute("SELECT 1")

                obj = cls.__new__(cls)
                obj._live_path = live_db_path
                obj._tmp_path = tmp_path
                obj.conn = conn
                obj._insert_stmt = insert_stmt
                obj._insert_sql = insert_sql
                obj._is_writer = True
                return obj

            except sqlite3.OperationalError:
                # Could not get a write txn on this tmp path; close and retry with a new unique path.
                try:
                    conn.close()
                finally:
                    pass
                if attempt <= 3:
                    time.sleep(0.05 * attempt)
                    continue
                raise

    @classmethod
    def open_reader(cls, live_db_path: str) -> "SequenceStore":
        """
        Open an immutable, read-only connection that never locks or blocks.
        """
        conn = sqlite3.connect(
            f"file:{live_db_path}?mode=ro&immutable=1&cache=private",
            uri=True,
            check_same_thread=False,
            timeout=0.0,
        )
        obj = cls.__new__(cls)
        obj._live_path = live_db_path
        obj._tmp_path = None
        obj.conn = conn
        obj._insert_stmt = None
        obj._insert_sql = None
        obj._is_writer = False
        return obj

    def close_publish(self) -> None:
        """
        Writers only: commit, optimize, close, then atomically replace the live DB.
        """
        if not getattr(self, "_is_writer", False):
            raise RuntimeError("close_publish() is only valid on a writer store")

        cur = self.conn.cursor()
        try:
            cur.execute("COMMIT;")
        except sqlite3.OperationalError:
            pass
        try:
            cur.execute("PRAGMA optimize;")
        except Exception:
            pass

        self.close()

        # Atomic publish
        assert self._tmp_path is not None
        os.replace(self._tmp_path, self._live_path)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def insert_rows(
        self,
        rows: Sequence[Tuple[int, int, int, bytes, int, int, Optional[bytes]]],
    ) -> None:
        """
        Fast batched insert.
        Each row: (gid, len, hash8, seq_blob, shard, local_idx, header_blob)
        """
        if not getattr(self, "_is_writer", False):
            raise RuntimeError("insert_rows() is only valid on a writer store")
        if not rows:
            return
        cur = self.conn.cursor()
        cur.executemany(self._insert_sql, rows)

    # lru caching beneficial if there are multiple queries hitting the same gid from query file?
    @functools.lru_cache(maxsize=32768)
    def get_seq(self, gid: int) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT seq FROM sequences WHERE gid=?", (int(gid),))
        row = cur.fetchone()
        return self.decode_seq(row[0]) if row else None

    def get_header_len(self, gid: int) -> Tuple[Optional[str], Optional[int]]:
        cur = self.conn.cursor()
        cur.execute("SELECT header, len FROM sequences WHERE gid=?", (int(gid),))
        row = cur.fetchone()
        if not row:
            return (None, None)
        header_blob, length = row
        return (self.decode_header(header_blob), int(length))

    def get_many_header_len(
        self, gids: Iterable[int]
    ) -> List[Tuple[int, Optional[str], Optional[int]]]:
        gids_list = [int(g) for g in gids]
        if not gids_list:
            return []
        if len(gids_list) <= 1000:
            qmarks = ",".join("?" for _ in gids_list)
            sql = f"SELECT gid, header, len FROM sequences WHERE gid IN ({qmarks})"
            cur = self.conn.cursor()
            cur.execute(sql, gids_list)
            found = {
                int(gid_): (self.decode_header(hdr_blob), int(len_val))
                for gid_, hdr_blob, len_val in cur.fetchall()
            }
            return [(g, *(found.get(g) or (None, None))) for g in gids_list]
        cur = self.conn.cursor()
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS gids_tmp(gid INTEGER PRIMARY KEY)")
        cur.execute("DELETE FROM gids_tmp;")
        cur.executemany(
            "INSERT INTO gids_tmp(gid) VALUES(?)", [(g,) for g in gids_list]
        )
        cur.execute(
            "SELECT s.gid, s.header, s.len FROM sequences s JOIN gids_tmp t ON s.gid=t.gid"
        )
        found = {
            int(gid_): (self.decode_header(hdr_blob), int(len_val))
            for gid_, hdr_blob, len_val in cur.fetchall()
        }
        cur.execute("DELETE FROM gids_tmp;")
        return [(g, *(found.get(g) or (None, None))) for g in gids_list]

    def get_many_meta(
        self, gids: Iterable[int]
    ) -> List[Tuple[int, Optional[str], Optional[int], Optional[int]]]:
        """
        Batched fetch of gid, header, len, and hash8.
        Returns a list of (gid, header, length, hash8) tuples.
        """
        gids_list = [int(g) for g in gids]
        if not gids_list:
            return []

        # temp table for large lists
        cur = self.conn.cursor()
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS gids_tmp(gid INTEGER PRIMARY KEY)")
        cur.execute("DELETE FROM gids_tmp;")

        # Use a transaction for the bulk insert into the temporary table
        with self.conn:
            cur.executemany(
                "INSERT INTO gids_tmp(gid) VALUES(?)", [(g,) for g in gids_list]
            )

        # Perform the join to retrieve all metadata in one go
        cur.execute(
            "SELECT s.gid, s.header, s.len, s.hash8 FROM sequences s JOIN gids_tmp t ON s.gid=t.gid"
        )

        results = [
            (gid, self.decode_header(hdr_blob), length, h8)
            for gid, hdr_blob, length, h8 in cur.fetchall()
        ]
        cur.execute("DROP TABLE gids_tmp;")
        return results

    def get_gids_by_length_range(
        self, min_len: Optional[int], max_len: Optional[int]
    ) -> List[int]:
        """Return all gids whose sequence length is within [min_len, max_len].
        Uses SQLite index on len for speed."""
        lo = 0 if min_len is None else int(min_len)
        hi = 2**31 - 1 if max_len is None else int(max_len)
        cur = self.conn.cursor()
        cur.execute("SELECT gid FROM sequences WHERE len BETWEEN ? AND ?", (lo, hi))
        return [int(r[0]) for r in cur.fetchall()]

    @staticmethod
    def hash8(seq: str) -> int:
        h = hashlib.sha1(seq.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "little", signed=True)

    @staticmethod
    def encode_seq(seq: str, use_zstd: bool) -> bytes:
        if not use_zstd:
            return b"\x00" + seq.encode("utf-8")
        try:
            import zstandard as zstd
        except Exception:
            return b"\x00" + seq.encode("utf-8")
        c = zstd.ZstdCompressor(level=10)
        return b"\x01" + c.compress(seq.encode("utf-8"))

    @staticmethod
    def decode_seq(blob: bytes) -> str:
        if not blob:
            return ""
        flag = blob[0]
        payload = blob[1:]
        if flag == 0:
            return payload.decode("utf-8")
        if flag == 1:
            import zstandard as zstd

            return zstd.ZstdDecompressor().decompress(payload).decode("utf-8")
        raise ValueError("Unknown sequence blob encoding flag")

    @staticmethod
    def encode_header(header: Optional[str], use_zstd: bool) -> Optional[bytes]:
        if header is None:
            return None
        if not use_zstd:
            return b"\x00" + header.encode("utf-8")
        try:
            import zstandard as zstd
        except Exception:
            return b"\x00" + header.encode("utf-8")
        c = zstd.ZstdCompressor(level=10)
        return b"\x01" + c.compress(header.encode("utf-8"))

    @staticmethod
    def decode_header(blob: Optional[bytes]) -> Optional[str]:
        if not blob:
            return None
        flag = blob[0]
        payload = blob[1:]
        if flag == 0:
            return payload.decode("utf-8")
        if flag == 1:
            import zstandard as zstd

            return zstd.ZstdDecompressor().decompress(payload).decode("utf-8")
        raise ValueError("Unknown header blob encoding flag")
