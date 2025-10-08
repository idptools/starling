#!/usr/bin/env python
"""FASTA pre-tokenization CLI (moved from starling.inference.starling_pretokenize)

Usage:
  starling-pretokenize -o out file1.fasta file2.fasta
  starling-pretokenize --sequences fasta_list.txt -o out --combined

Outputs:
  Per FASTA: <basename>.tokens.pt  (list[ {header, sequence, tokenized} ])
  Combined:  <prefix>.pt with { files: [...], entries: [...] }
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import torch
from protfasta import read_fasta
from tqdm.auto import tqdm

from starling.data.tokenizer import StarlingTokenizer


def tokenize_fasta(path: str, tokenizer: StarlingTokenizer) -> List[dict]:
    records = read_fasta(path)
    out: List[dict] = []
    for header, seq in records.items():
        seq_up = seq.strip().upper()
        tokens = tokenizer.encode(seq_up)
        out.append({"header": header, "sequence": seq_up, "tokenized": tokens})
    return out


def save_pt(obj, path: str):
    torch.save(obj, path)
    n = len(obj) if isinstance(obj, list) else len(obj.get("entries", []))
    print(f"Wrote {path} ({n} entries)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pre-tokenize FASTA files")
    p.add_argument("fastas", nargs="*", help="Input FASTA files")
    p.add_argument("--output", "-o", required=True, help="Output directory")
    p.add_argument("--combined", action="store_true", help="Write single combined file")
    p.add_argument("--prefix", default="pretokenized", help="Combined output prefix")
    p.add_argument(
        "--sequences", help="Text file with absolute FASTA paths (one per line)"
    )
    p.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    return p.parse_args(argv)


def _process_single(path: str) -> List[dict]:
    tok = StarlingTokenizer()
    return tokenize_fasta(path, tok)


def _gather_paths(args) -> List[str]:
    if args.sequences:
        with open(args.sequences) as f:
            paths = [ln.strip() for ln in f if ln.strip()]
    else:
        paths = list(args.fastas)
    if not paths:
        raise SystemExit("No FASTA inputs provided.")
    return paths


def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(args.output, exist_ok=True)
    paths = _gather_paths(args)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"Error: missing FASTA paths: {missing}", file=sys.stderr)
        return 1
    workers = max(1, int(args.workers))
    results: List[tuple[str, List[dict]]] = []
    show = not args.no_progress
    if workers == 1:
        tok = StarlingTokenizer()
        iterable = paths if not show else tqdm(paths, desc="Tokenizing", unit="file")
        for pth in iterable:
            results.append((pth, tokenize_fasta(pth, tok)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_map = {ex.submit(_process_single, p): p for p in paths}
            iterator = as_completed(future_map)
            if show:
                iterator = tqdm(
                    iterator, total=len(paths), desc="Tokenizing", unit="file"
                )
            for fut in iterator:
                pth = future_map[fut]
                try:
                    entries = fut.result()
                except Exception as e:
                    print(f"Failed processing {pth}: {e}", file=sys.stderr)
                    return 1
                results.append((pth, entries))
        order = {p: i for i, p in enumerate(paths)}
        results.sort(key=lambda x: order[x[0]])
    if args.combined:
        combined: List[dict] = []
        for _, entries in results:
            combined.extend(entries)
        payload = {"files": [os.path.abspath(f) for f in paths], "entries": combined}
        out_path = os.path.join(args.output, f"{args.prefix}.pt")
        save_pt(payload, out_path)
    else:
        saving_iter = results if not show else tqdm(results, desc="Saving", unit="file")
        for pth, entries in saving_iter:
            base = os.path.splitext(os.path.basename(pth))[0]
            out_path = os.path.join(args.output, f"{base}.tokens.pt")
            save_pt(entries, out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
