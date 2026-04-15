Utilities
=========

The :mod:`starling.utilities` module provides helper functions for path
management, device selection, ensemble file I/O, and miscellaneous
structural operations. Most of these are used internally by the higher-level
API, but they are useful when building custom workflows.

File and Path Helpers
---------------------

.. code-block:: python

   from starling.utilities import (
       get_data,
       fix_ref_to_home,
       check_file_exists,
       remove_extension,
   )

   # Access bundled data files
   path = get_data("some_bundled_file.dat")

   # Expand ~ in a path
   full_path = fix_ref_to_home("~/projects/starling/output")

   # Check whether a file exists (raises on missing)
   check_file_exists("ensemble.starling")

   # Strip the extension from a filename
   stem = remove_extension("ensemble.starling")  # "ensemble"

Device Management
-----------------

:func:`~starling.utilities.check_device` normalises a user-supplied device
string into a :class:`torch.device` object, automatically selecting the best
available accelerator when the input is ``"gpu"``:

.. code-block:: python

   from starling.utilities import check_device

   # Automatically pick GPU (CUDA or MPS) if available, else CPU
   device = check_device("gpu")

   # Force a specific CUDA device
   device = check_device("cuda:1")

   # Use CPU explicitly
   device = check_device("cpu")

On macOS, :func:`~starling.utilities.get_macOS_version` is used internally
to determine MPS availability.

Ensemble File I/O
-----------------

STARLING archives use a custom ``.starling`` format that bundles distance maps,
metadata, and (optionally) 3D trajectories. The low-level readers and writers
are exposed for advanced use:

.. code-block:: python

   from starling.utilities import write_starling_ensemble, read_starling_ensemble

   # Save an ensemble object to disk
   write_starling_ensemble(
       ensemble_object,
       "my_ensemble",
       compress=True,
       reduce_precision=2,           # decimal places to keep
       compression_algorithm="lzma", # or "gzip"
       verbose=True,
   )

   # Load the archive back
   data = read_starling_ensemble("my_ensemble.starling")

In most cases you should prefer the higher-level
:meth:`~starling.structure.ensemble.Ensemble.save` and
:func:`~starling.structure.ensemble.load_ensemble` methods, which wrap
these functions and handle trajectory serialisation automatically.

Structural Helpers
------------------

:func:`~starling.utilities.helix_dm` generates a reference Cα distance map for
an ideal α-helix of a given length. This is used internally by
:class:`~starling.inference.constraints.HelicityConstraint` but can be useful
for custom analyses:

.. code-block:: python

   from starling.utilities import helix_dm

   dm = helix_dm(20)  # 20-residue ideal helix distance map

See Also
--------

* :doc:`ensemble` – Working with ensembles at a higher level.
* :doc:`ensemble_generation` – The main generation API.
* :mod:`starling.utilities` – Full API reference.
