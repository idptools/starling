Similarity Search
=================

STARLING pairs its ensemble-aware sequence encoder with FAISS to deliver fast
nearest-neighbour searches over millions of protein sequences. Use this guide to
build a reusable index, embed queries, and interpret the rich filtering options
exposed by :class:`starling.search.SearchEngine`.

Prerequisites
-------------

* Install the ``search-gpu`` extra (``pip install idptools-starling[search-gpu]``)
  when working with CUDA-enabled FAISS; the standard installation already
  includes ``faiss-cpu``.
* Prepare tokenised sequence shards with ``starling-pretokenize`` or your own
  data pipeline. Each shard stores per-residue latent codes used during indexing.
* The first invocation of ``starling-search query`` (or :meth:`SearchEngine.load`
  with ``index_path="default"``) downloads the pre-built UniRef50 reference
  index. The archive is cached under ``~/.starling_search`` so subsequent runs
  reuse the local copy without additional downloads.

Building indexes
----------------

.. code-block:: python

   from starling.search import IndexBuilder

   builder = IndexBuilder(
       root="/data/starling_corpus",
       metric="cosine",
       verbose=True,
       shard_id_regex=r"shard_(\d+)\.h5",
   )

   builder.build_index(
       index_path="/data/indexes/uniref50.faiss",
       tokens_dir="/data/starling_corpus/tokens",
       sample_size=1_000_000,
       nlist=32768,
       m=64,
       nbits=8,
       use_gpu=True,
       gpu_fp16_lut=True,
       compress_sequences=True,
   )

``IndexBuilder`` writes a FAISS index alongside a SQLite-backed
:class:`~starling.search.store.SequenceStore` that retains the original headers,
lengths, and (optionally) sequences for reranking. For a shell-first workflow
use the bundled CLI:

.. code-block:: bash

   starling-search build \
       --root /data/starling_corpus \
       --tokens /data/starling_corpus/tokens \
       --index /data/indexes/uniref50.faiss \
       --sample-size 1000000 \
       --nlist 32768 --m 64 --nbits 8 \
       --use-gpu --gpu-device 0 --opq

Querying from Python
--------------------

.. code-block:: python

   import torch
   from starling.search import SearchEngine
   from starling.inference.generation import sequence_encoder_backend

   index_path = "/data/indexes/uniref50.faiss"
   engine = SearchEngine.load(index_path, metric="cosine", verbose=True)

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
``(score, gid, header, length)``. When ``return_similarity`` is ``True`` and the
metric is cosine, ``score`` is the similarity value; otherwise it contains the
raw FAISS distance.

Filtering and reranking
-----------------------

The search engine composes several filter primitives:

* **Length gating** - quickly discard candidates outside a residue range before
  reranking.
* **Exact match suppression** - exclude entries that exactly match the input
  sequence when ``exclude_exact=True``.
* **Identity threshold** - use ``sequence_identity_max`` with
  ``identity_denominator`` (``"query"``, ``"target"``, ``"max"``, ``"min"``, or
  ``"avg"``) to control how similarity is computed.
* **Metric clamps** - ``max_cosine_similarity`` or ``min_l2_distance`` allow
  coarse filtering straight out of FAISS.
* **Reranking** - set ``rerank=True`` to re-embed the top candidates with the
  full encoder for exact scoring. ``rerank_device`` and ``rerank_batch_size``
  manage resources, while ``rerank_ionic_strength`` lets you score under
  different ionic-strength conditions.

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
       --rerank --rerank-device cuda:0 \
       --out search_results.csv

Results can be saved as CSV/JSONL and optionally exported to FASTA for
inspection. Passing ``--index default`` downloads the reference STARLING index
if available and caches it locally.

Next steps
----------

* :doc:`usage/sequence_encoder` - extract embeddings for downstream analysis or
  custom similarity metrics.
* :doc:`usage/ensemble_generation` - generate new ensembles for promising hits.
* :doc:`usage/possible_issues` - diagnose FAISS installation or GPU related
  problems.
