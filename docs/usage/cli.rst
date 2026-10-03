Command Line Interface
======================

STARLING ships with a collection of console scripts that cover ensemble
generation, performance benchmarking, format conversion, and similarity search.
This page summarises the most common commands and how they fit into an
end-to-end workflow.

``starling``
------------

Generate conformational ensembles directly from the shell. The CLI mirrors the
:func:`starling.generate` signature and accepts sequences, FASTA/TSV files, or
lists of sequences.

.. code-block:: bash

   starling my_sequences.fasta \
       -c 200 \
       --ionic_strength 150 \
       --return_structures \
       --output_directory outputs \
	   -r
	   

Key options:

* ``-c`` / ``--conformations`` - number of conformers to sample (default 400)
* ``-r``/``--return_structures`` – Flag which, if provided, means STARLING returns 3D coordinates alongside distance maps as a .pdb file (single topology file) and a .xtc file (compressed trajectory format).
* ``--ionic_strength`` - choose 20, 150, or 300 mM solvent environments
* ``--steps`` - diffusion steps for the sampler (default 25)
* ``--device`` - force CPU, CUDA (``cuda:0``), or Apple MPS
* ``--num-cpus`` / ``--num-mds-init`` - control MDS reconstruction throughput
* ``--outname`` - override the output prefix when providing a single sequence
* ``--remove-errors`` - discard physically impossible conformations and regenerate replacements (see below)


Outputs live under the requested directory and includes ``.starling`` archives
plus optional PDB/XTC trajectories when ``--return_structures`` is set.

Removing erroneous conformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Occasionally STARLING produces a conformation containing physically impossible
inter-residue distances - two residues further apart than a fully extended chain
between them would allow. Passing ``--remove-errors`` discards these and
generates replacements, so the number of conformations you get back still
matches ``-c``:

.. code-block:: bash

    starling my_sequence.fasta -c 200 -r --remove-errors -o outputs

Screening happens in two stages:

1. The raw STARLING distance maps are screened with
   :meth:`~starling.structure.ensemble.Ensemble.check_for_errors`.
2. When ``-r``/``--return_structures`` is also set, the reconstructed 3D
   conformations are screened with
   :meth:`~starling.structure.ensemble.Ensemble.check_for_errors_trajectory`,
   which catches artefacts introduced by the MDS embedding itself.

The cheap distance-map screen runs first, so 3D reconstruction is only spent on
conformations that already look plausible. Discarded conformations are replaced
in further sampling rounds until the requested number of clean conformations has
been accumulated. Because all of this happens *before* anything is written, the
``.xtc``, ``.pdb`` and ``.starling`` outputs contain only conformations that
passed both screens.

.. note::

   Without ``-r``, only the distance-map screen can run - there is no
   reconstructed trajectory to inspect - but the ``.starling`` archive still
   ends up free of erroneous distance maps.

   If STARLING cannot accumulate enough clean conformations after 10 sampling
   rounds it raises a ``RuntimeError`` rather than silently returning fewer
   conformations than requested. In practice this indicates something is wrong
   with the sequence or the model, so it is worth re-running without the flag to
   inspect the raw output.

The related ``--remove-errors`` flag on ``starling2xtc`` / ``starling2pdb``
(:ref:`below <removing-erroneous-frames>`) is different: those operate on an
already-generated ``.starling`` archive, so they can only *remove* bad frames -
they cannot generate replacements.

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
     - Convert a STARLING archive into a multi-model PDB trajectory. Pass
       ``--remove-errors`` to drop physically impossible frames first (see below).
   * - ``starling2xtc``
     - Export a topology PDB paired with an XTC trajectory (reconstructs
       coordinates if necessary). Pass ``--remove-errors`` to drop physically
       impossible frames first (see below).
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

.. _removing-erroneous-frames:

Removing erroneous frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``starling2pdb`` and ``starling2xtc`` accept a ``--remove-errors`` flag. When
set, the reconstructed 3D trajectory is scanned for frames containing
physically impossible inter-residue distances (a pair of residues separated by
``|i - j|`` positions in the sequence cannot be further apart than
``|i - j|`` bond lengths), and any such frames are removed *before* the
trajectory is written to disk. This is helpful when a particular sequence is
badly behaved and the SMACOF reconstruction occasionally produces unphysical
geometry.

.. code-block:: bash

   starling2xtc my_ensemble.starling -o cleaned.xtc --remove-errors

The equivalent distance-map check (operating on the raw STARLING distance maps
rather than the reconstructed coordinates) is available via ``starling2starling
--error-check --remove-errors`` and the
:meth:`starling.structure.ensemble.Ensemble.check_for_errors` method.

Search tooling
--------------

STARLING bundles a FAISS-based similarity search stack to explore large
sequence collections.

* ``starling-pretokenize`` - preprocess a FASTA into shard-wise token files that
  power fast index construction.
* ``starling-search build`` - create a FAISS index from a pretokenized corpus.
* ``starling-search query`` - embed query sequences with the STARLING encoder and
  retrieve the nearest neighbours with optional reranking.

See :doc:`search` for a complete walkthrough of building and querying
indexes as well as the Python API.

Training tools (advanced)
-------------------------

These entry points are used for model development and retraining, and are not
needed to generate or analyse ensembles.

* ``starling-vae-train`` - train the VAE encoder/decoder.
* ``starling-ddpm-train`` - train the diffusion model.

.. important::

   The training tools require Hydra, OmegaConf and Weights & Biases, which are
   **not** installed by default. Install the ``train`` extra first, otherwise
   these commands fail at startup with a ``ModuleNotFoundError``:

   .. code-block:: bash

       pip install "idptools-starling[train]"

   From a local clone, use ``pip install ".[train]"``. See
   :ref:`training-dependencies` for details.

Both commands are Hydra applications, so configuration is supplied through the
YAML files in ``starling/configs/`` and can be overridden on the command line,
for example:

.. code-block:: bash

    starling-ddpm-train trainer.max_epochs=100

Tips
----

* Use ``--info`` or ``--version`` with ``starling`` to inspect default model
  paths without launching a generation run.
* All CLI commands respect the ``CUDA_VISIBLE_DEVICES`` environment variable—set
  it ahead of time for multi-GPU systems.
* Each converter lazily reconstructs structures the first time they are needed
  and caches the trajectory in the ``.starling`` archive for subsequent use.
