Similarity Search
=================

STARLING pairs its ensemble-aware sequence encoder with FAISS to deliver fast
nearest-neighbour searches over millions of protein sequences. Use this guide to
build a reusable index, embed queries, and interpret the rich filtering options
exposed by :class:`~starling.search.search_engine.SearchEngine`.

Prerequisites
-------------

* The standard installation includes ``faiss-cpu``. GPU FAISS is not available
  as a pip extra; install it from conda instead (see
  :ref:`gpu-installation-cuda` in :doc:`installation`).
* Building your own index needs two inputs per shard: a
  ``sequence_features.pt`` file holding one embedding per sequence, with the
  sequence headers, and a tokens file made with ``starling-pretokenize`` (see
  `Shard layout`_ below).
* The first invocation of ``starling-search query`` with ``--index default``
  (the default) downloads the pre-built UniRef50 reference index. The files are
  cached under ``~/.starling_search`` so subsequent runs reuse the local copy
  without additional downloads. :meth:`SearchEngine.load
  <starling.search.search_engine.SearchEngine.load>` never downloads anything: from Python,
  fetch the default index with :func:`starling.configs.ensure_search_artifacts`
  and pass its path:

  .. code-block:: python

     from starling.configs import ensure_search_artifacts
     from starling.search import SearchEngine

     index_path, seqstore_path, manifest_path = ensure_search_artifacts()
     engine = SearchEngine.load(index_path)

Building indexes
----------------

.. code-block:: python

   from starling.search import IndexBuilder

   builder = IndexBuilder(
       root="/data/starling_corpus",
       metric="cosine",
       verbose=True,
       shard_id_regex=r"my_shard_(\d+)",  # default: r"uniref50_idrs_only_(\d{6})"
   )

   builder.build_index(
       index_path="/data/indexes/uniref50.faiss",
       tokens_dir="/data/starling_corpus/tokens",
       sample_size=1_000_000,
       nlist=32768,
       m=64,               # number of PQ subquantizers
       nbits=8,            # bits per subquantizer
       gpu_fp16_lut=True,  # use_gpu defaults to None: use a GPU only if FAISS reports one
       compress_sequences=True,
   )

``IndexBuilder`` builds an OPQ+IVF-PQ index (OPQ is on by default; pass
``use_opq=False`` or ``--no-opq`` for plain IVF-PQ) and writes it alongside a
manifest and a SQLite-backed :class:`~starling.search.store.SequenceStore` that
retains the original headers, lengths, and (optionally compressed) sequences
for filtering and reranking. For a shell-first workflow use the bundled CLI
(the GPU is used automatically if FAISS reports one; ``--no-gpu`` forces CPU):

.. code-block:: bash

   starling-search build \
       --root /data/starling_corpus \
       --tokens /data/starling_corpus/tokens \
       --index /data/indexes/uniref50.faiss \
       --sample-size 1000000 \
       --nlist 32768 --m 64 --nbits 8 \
       --shard-regex 'my_shard_(\d+)'

Shard layout
~~~~~~~~~~~~

Feature shards are found recursively under ``root`` as
``sequence_features.pt`` files. ``shard_id_regex`` (default
``uniref50_idrs_only_(\d{6})``) is searched for in each file's path: its first
capture group is the numeric shard id (ids must be unique), and the directory
name in which it matches is the shard directory name. The tokens for each
shard must be in ``<tokens_dir>/<shard directory name>.tokens.pt``,
which is exactly what ``starling-pretokenize`` writes for a FASTA file named
``<shard directory name>.fasta``:

.. code-block:: text

   /data/starling_corpus/
   ├── my_shard_000000/
   │   └── sequence_features.pt
   ├── my_shard_000001/
   │   └── sequence_features.pt
   └── tokens/
       ├── my_shard_000000.tokens.pt
       └── my_shard_000001.tokens.pt

Embeddings are paired with sequences by header, not by position, so each
feature shard must carry headers that match its tokens file one-to-one
(supported layouts are ``{"features": {header: vector}}``,
``{"features": tensor, "headers": [...]}``, ``{header: vector}``, or a list of
``{"header": ..., "embedding": vector}`` records). Every tokens file is checked
before training starts, so a bad tokens directory fails quickly.

Querying from Python
--------------------

.. code-block:: python

   import torch
   from starling.search import SearchEngine
   from starling.inference.generation import sequence_encoder_backend

   index_path = "/data/indexes/uniref50.faiss"
   # metric defaults to the index's own metric; if given, it must match
   engine = SearchEngine.load(index_path, verbose=True)

   sequences = {
       "hnrnpa1": "GGRSGRGGGFGGGGGGGGGY...",
       "synuclein": "MDVFMKGLSKAKEGVVAAAEKTKQGVAE...",
   }

   embeddings = sequence_encoder_backend(
       sequence_dict=sequences,
       device="cuda:0",
       batch_size=64,
       ionic_strength=150,
       aggregate=True,
       output_directory=None,
   )

   queries = torch.stack([embeddings[name] for name in sequences]).float()
   queries = torch.nn.functional.normalize(queries, dim=1)

   results = engine.search(
       queries=queries,
       k=50,
       nprobe=128,
       return_similarity=True,
       query_sequences=list(sequences.values()),
       exclude_exact=True,
       length_min=50,
       length_max=600,
       max_cosine_similarity=0.95,
       rerank=True,
       rerank_device="cuda:0",
   )

Each row in ``results`` is a list of tuples
``(score, gid, header, length)``. For a cosine index, ``score`` is the cosine
similarity when ``return_similarity`` is ``True`` and the cosine distance
(``1 - similarity``) otherwise; for an L2 index it is the Euclidean distance.
With ``rerank=True`` the scores are the exact rerank scores. If ``nprobe`` is
not given, the probe count stored with the index is used.

Filtering and reranking
-----------------------

The search engine composes several filter primitives:

* **Length gating** - quickly discard candidates outside a residue range before
  reranking.
* **Exact match suppression** - exclude entries that exactly match the input
  sequence when ``exclude_exact=True`` (requires ``query_sequences``).
* **Identity threshold** - use ``sequence_identity_max`` (inclusive, 0-1) with
  ``identity_denominator`` (``"query"``, ``"target"``, ``"max"``, ``"min"``, or
  ``"avg"``) to control how identity is computed (requires
  ``query_sequences``).
* **Metric clamps** - ``max_cosine_similarity`` (cosine indexes) or
  ``min_l2_distance`` (L2 indexes), both inclusive. They are applied to the
  FAISS scores, or to the exact rerank scores when ``rerank=True``.
* **Reranking** - set ``rerank=True`` (off by default in Python, on by default
  in ``starling-search query``) to re-embed the top candidates with the full
  encoder for exact scoring. When ``query_sequences`` is given, the queries are
  re-encoded too, so queries and candidates are compared at the same ionic
  strength, ``rerank_ionic_strength`` (default 150 mM in Python; on the command
  line it defaults to ``--ionic-strength``). ``rerank_device`` and
  ``rerank_batch_size`` manage resources.

Command-line querying
---------------------

.. code-block:: bash

   starling-search query \
       --index /data/indexes/uniref50.faiss \
       --seq MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKM \
       --seq MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKT \
       --k 20 --nprobe 128 \
       --exclude-exact --sequence-identity-max 0.9 \
       --length-min 40 --length-max 800 \
       --rerank-device cuda:0 \
       --out search_results

Several query sequences can follow one ``--seq`` or the flag can be repeated,
as above. Reranking is on by default (``--no-rerank`` turns it off),
``--metric`` defaults to the index's own metric (a mismatch is an error), and
``--device`` is picked automatically (CUDA, then MPS, then CPU) unless given.
Results are written to ``<out>.csv`` (or ``<out>.jsonl`` with
``--out-format jsonl``; any extension on ``--out`` is replaced), together with
a FASTA file of the hits, ``<out>.fasta``. Passing ``--index default`` (the
default), or a path that does not exist, downloads the reference STARLING
index and caches it locally.

Next steps
----------

* :doc:`sequence_encoder` - extract embeddings for downstream analysis or
  custom similarity metrics.
* :doc:`ensemble_generation` - generate new ensembles for promising hits.
* :doc:`possible_issues` - diagnose FAISS installation or GPU related
  problems.
