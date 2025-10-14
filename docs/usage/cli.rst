Command Line Interface
======================

STARLING ships with a collection of console scripts that cover ensemble
generation, performance benchmarking, format conversion, and similarity search.
This page summarises the most common commands and how they fit into an
end-to-end workflow.

``starling``
-----------

Generate conformational ensembles directly from the shell. The CLI mirrors the
:func:`starling.generate` signature and accepts sequences, FASTA/TSV files, or
lists of sequences.

.. code-block:: bash

   starling my_sequences.fasta \
       -c 200 \
       --ionic_strength 150 \
       --return_structures \
       --output_directory outputs

Key options:

* ``-c`` / ``--conformations`` - number of conformers to sample (default 200)
* ``--ionic_strength`` - choose 20, 150, or 300 mM solvent environments
* ``--steps`` - diffusion steps for the sampler (default 25)
* ``--device`` - force CPU, CUDA (``cuda:0``), or Apple MPS
* ``--num-cpus`` / ``--num-mds-init`` - control MDS reconstruction throughput
* ``--outname`` - override the output prefix when providing a single sequence

Outputs live under the requested directory and include ``.starling`` archives
plus optional PDB/XTC trajectories when ``--return_structures`` is set.

``starling-benchmark``
----------------------

Profile model throughput under different diffusion steps, conformer counts, and
hardware options.

.. code-block:: bash

   starling-benchmark --device cuda:0 --batch-size 64 --steps 30 --single-run 500

The command records runtime and radius-of-gyration measurements to CSV files for
later analysis.

Conversion utilities
--------------------

All converters operate on ``.starling`` archives created by the generator.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Purpose
   * - ``starling2pdb``
     - Convert a STARLING archive into a multi-model PDB trajectory.
   * - ``starling2xtc``
     - Export a topology PDB paired with an XTC trajectory (reconstructs
       coordinates if necessary).
   * - ``starling2numpy``
     - Dump raw distance maps to a Numpy ``.npy`` array for custom analyses.
   * - ``starling2sequence``
     - Print the amino-acid sequence associated with an archive.
   * - ``starling2info``
     - Display metadata such as version, conformer count, and default weights.
   * - ``starling2starling`` / ``numpy2starling`` / ``xtc2starling``
     - Regenerate archives from alternative representations.

By default outputs are written next to the source file; pass ``-o`` to choose a
new directory or filename prefix.

Search tooling
--------------

STARLING bundles a FAISS-based similarity search stack to explore large
sequence collections.

* ``starling-pretokenize`` - preprocess a FASTA into shard-wise token files that
  power fast index construction.
* ``starling-search build`` - create a FAISS index from a pretokenized corpus.
* ``starling-search query`` - embed query sequences with the STARLING encoder and
  retrieve the nearest neighbours with optional reranking.

See :doc:`usage/search` for a complete walkthrough of building and querying
indexes as well as the Python API.

Tips
----

* Use ``--info`` or ``--version`` with ``starling`` to inspect default model
  paths without launching a generation run.
* All CLI commands respect the ``CUDA_VISIBLE_DEVICES`` environment variable—set
  it ahead of time for multi-GPU systems.
* Each converter lazily reconstructs structures the first time they are needed
  and caches the trajectory in the ``.starling`` archive for subsequent use.
