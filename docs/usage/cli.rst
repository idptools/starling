Command Line Interface
======================

STARLING ships with a collection of console scripts that cover ensemble
generation, performance benchmarking, format conversion, and similarity search.
This page summarises the most common commands and how they fit into an
end-to-end workflow.

``starling``
------------

Generate conformational ensembles directly from the shell. The CLI mirrors the
:func:`starling.generate` signature and accepts a single sequence or a
FASTA/TSV/``.seq.in`` file.

.. code-block:: bash

   mkdir -p outputs
   starling my_sequences.fasta \
       -c 200 \
       --ionic_strength 150 \
       --output_directory outputs \
       -r

Key options:

* ``-c`` / ``--conformations`` - number of conformers to sample (default 400)
* ``-r``/``--return_structures`` – Flag which, if provided, means STARLING returns 3D coordinates alongside distance maps as a .pdb file (single topology file) and a .xtc file (compressed trajectory format). The structures are relaxed with Mpipi-GG by default (see below).
* ``--relax [standard|fast]`` / ``--no-relax`` - relax the ``-r`` structures
  with Mpipi-GG (``standard`` is the default whenever ``-r`` is given, and what
  ``--relax`` alone means); ``--relax fast`` runs a shorter MDS step before
  relaxing, and ``--no-relax`` keeps the raw reconstructed structures.
  ``--relax`` without ``-r`` is an error
* ``--compile-forces`` - fuse the CUDA thermalization forces for faster
  relaxation of large jobs (adds compilation time; needs ``-r``, relaxation and
  a CUDA device)
* ``--seed`` - random seed; the same seed reproduces the same ensembles on the
  same hardware and settings (default: unseeded)
* ``--ionic_strength`` - ionic strength in mM (default 150; STARLING was
  trained at 20, 150 and 300 mM)
* ``-s`` / ``--steps`` - sampler steps (default 30; DDIM warns below 10 steps,
  PLMS needs at least 20, and DDPM always uses all 1000 timesteps)
* ``--sampler`` - ``ddim`` (default), ``ddpm``, ``plms`` or ``dpmpp``
* ``-b`` / ``--batch_size`` - conformations sampled per batch (default 100)
* ``--device`` - force CPU, CUDA (``cuda:0``), or Apple MPS
* ``-o`` / ``--output_directory`` - where to write results (default ``.``; the
  directory must already exist)
* ``--outname`` - override the output prefix when providing a single sequence
* ``--remove-errors`` - discard physically impossible conformations and regenerate replacements (see below)
* ``--disable_progress_bar`` - hide both progress bars

Outputs live under the requested directory: ``<name>.starling`` archives plus
``<name>_STARLING.pdb`` and ``<name>_STARLING.xtc`` when ``--return_structures``
is set. ``<name>`` is the full sequence name (dots included); names containing
``/`` are rejected before sampling starts.

Relaxing structures
~~~~~~~~~~~~~~~~~~~

STARLING predicts distance maps, and ``-r`` reconstructs 3D structures from
them by classical MDS followed by weighted SMACOF. This gets the global shape
right but leaves bonds compressed and beads too close together, so by default
each structure is then relaxed with the Mpipi-GG force field while restrained
to its STARLING distance map: a FIRE energy minimization followed by 250 steps
of Langevin thermalization at 300 K. Local geometry ends up close to that of
Mpipi-GG simulations while global dimensions are essentially unchanged. Use
``--no-relax`` for the raw reconstructed structures.

``--relax fast`` computes the MDS starting point from only the eigenvectors it
needs and runs 100 rather than 300 SMACOF iterations before relaxing, which
makes the reconstruction about 4x faster for long sequences. The relaxed
ensembles match the standard ones in fit to the distance maps, Rg, bond
lengths and closest contacts, though individual conformations differ (see
:ref:`relaxing-structures`).

Structures built later from an archive (for example by ``starling2xtc``) are
not relaxed; see :ref:`relaxing-existing-ensembles` for relaxing an existing
ensemble from Python.

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
2. When ``-r``/``--return_structures`` is also set, the reconstructed (and,
   by default, relaxed) 3D conformations are screened with
   :meth:`~starling.structure.ensemble.Ensemble.check_for_errors_trajectory`,
   which catches artefacts introduced by the MDS embedding itself. Relaxation
   runs before this screen, so conformations broken by MDS are repaired rather
   than discarded.

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
   * - ``starling2starling``
     - Error-check an archive (``--error-check``); with ``--remove-errors``
       (which implies ``--error-check``) write ``fixed_<name>.starling`` with
       the erroneous conformations removed. Nothing is written if no errors
       are found.
   * - ``numpy2starling`` / ``xtc2starling``
     - Build archives from a NumPy distance-map array or an XTC trajectory.
       ``numpy2starling`` checks that an attached XTC has one frame per
       distance map; ``xtc2starling`` keeps the trajectory's coordinates as
       the archive's structures.

By default outputs are written to the current directory and named after the
input file; pass ``-o`` with an existing directory (the output keeps the
input's name) or a filename prefix. ``starling2sequence`` has no ``-o``; it
prints to stdout, so redirect it with ``>``. The converters never re-save the
input archive, and structures that ``starling2pdb``/``starling2xtc``
reconstruct from the distance maps are not relaxed.

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

   starling2xtc my_ensemble.starling -o cleaned --remove-errors

This writes ``cleaned.pdb`` and ``cleaned.xtc``. The equivalent distance-map
check (operating on the raw STARLING distance maps rather than the
reconstructed coordinates) is available via ``starling2starling
--remove-errors`` and the
:meth:`starling.structure.ensemble.Ensemble.check_for_errors` method:

.. code-block:: bash

   # writes ./fixed_my_ensemble.starling if any erroneous conformations are found
   starling2starling my_ensemble.starling --remove-errors

   # ...or into an existing directory
   starling2starling my_ensemble.starling --remove-errors -o cleaned_archives

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

   The training tools require Hydra, OmegaConf, Weights & Biases, WebDataset
   and tabulate, which are **not** installed by default. Install the ``train`` extra first, otherwise
   these commands fail at startup with a ``ModuleNotFoundError``:

   .. code-block:: bash

       pip install "idptools-starling[train]"

   From a local clone, use ``pip install ".[train]"``. See
   :ref:`training-dependencies` for details.

Both commands are Hydra applications, so configuration is supplied through the
YAML files in ``starling/configs/`` and can be overridden on the command line,
for example:

.. code-block:: bash

    starling-ddpm-train trainer.num_epochs=100

Diffusion training resumes automatically from ``last.ckpt`` if an earlier run
left one in the output directory. Only the WebDataset (tar shard) dataloaders
are supported; the ``h5`` dataloader type raises an error.

Tips
----

* Use ``--info`` or ``--version`` with ``starling`` to inspect default model
  paths without launching a generation run.
* All CLI commands respect the ``CUDA_VISIBLE_DEVICES`` environment variable—set
  it ahead of time for multi-GPU systems.
* ``starling2pdb`` and ``starling2xtc`` reconstruct (unrelaxed) structures when
  the archive has none, but they do not save them back into the ``.starling``
  archive, so the reconstruction is repeated on every call. To keep structures
  with an archive, generate with ``-r`` or save an ensemble that has structures
  from Python.
