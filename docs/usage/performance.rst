Performance Optimization
========================

STARLING can be accelerated through PyTorch compilation to achieve faster
sampling throughput on repeated runs. Use this page to understand compilation
options and optimize your workflow for high-throughput applications.

Overview
--------

PyTorch's ``torch.compile`` infrastructure can dramatically speed up ensemble
generation by:

1. **Optimizing computation graphs** – reducing Python overhead
2. **Fusing operations** – combining multiple kernels into efficient sequences
3. **Caching compiled models** – amortizing compilation cost across runs

STARLING caches compiled models between calls, so the compilation overhead is
paid once and benefits all subsequent sampling jobs.

Basic Usage
-----------

Enable compilation with :func:`starling.set_compilation_options`:

.. code-block:: python

   import starling

   # Enable compilation with default settings
   starling.set_compilation_options(enabled=True)

   # Generate ensembles - first call compiles, subsequent calls are faster
   for sequence in sequences:
       ensemble = starling.generate(sequence, conformations=200)

The first call will take longer due to compilation, but subsequent calls will
be significantly faster.

Compilation Modes
-----------------

PyTorch supports several compilation modes that trade off compilation time for
runtime performance:

``"default"``
   Balanced mode suitable for most cases. Good speedup with reasonable
   compilation time.

``"reduce-overhead"``
   **Recommended for STARLING.** Optimizes for minimal Python overhead and
   fast execution. Best for repeated sampling runs.

   .. code-block:: python

      starling.set_compilation_options(
          enabled=True,
          mode="reduce-overhead"
      )

``"max-autotune"``
   Extensive tuning for maximum performance. Takes longer to compile but
   produces the fastest code. Use for production workloads with fixed
   sequences.

   .. code-block:: python

      starling.set_compilation_options(
          enabled=True,
          mode="max-autotune"
      )

Backend Selection
-----------------

The compilation backend determines how PyTorch optimizes and executes your
models:

``"inductor"`` (default)
   Modern TorchInductor backend with excellent performance on both CPU and GPU.
   Supports most PyTorch operations and provides strong speedups.

   .. code-block:: python

      starling.set_compilation_options(
          enabled=True,
          backend="inductor"
      )

``"cudagraphs"`` (GPU only)
   Captures and replays entire CUDA execution graphs. Can provide additional
   speedup on GPUs for fixed-shape workloads.

Advanced Options
----------------

Full Configuration Example
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import starling

   starling.set_compilation_options(
       enabled=True,
       mode="reduce-overhead",
       backend="inductor",
       fullgraph=False,          # Allow graph breaks
       dynamic=False,            # Fixed tensor shapes
       options={
           "triton.cudagraphs": True,  # Backend-specific options
       }
   )

Common Options
^^^^^^^^^^^^^^

``fullgraph`` : bool, default False
   If ``True``, requires the entire model to compile as a single graph.
   Compilation may fail if the model contains unsupported operations.
   Set to ``False`` to allow graph breaks.

``dynamic`` : bool, default None
   Controls dynamic shape support. Set to ``False`` for fixed-shape workloads
   (faster) or ``True`` for variable-shape inputs.

``options`` : dict, optional
   Backend-specific configuration. See PyTorch documentation for details.

Disabling Compilation
---------------------

To restore eager execution mode:

.. code-block:: python

   starling.set_compilation_options(enabled=False)

This is useful for debugging or when compilation is causing issues.

Performance Tips
----------------

1. **Warm-up runs**: The first generation after enabling compilation will be
   slower due to compilation overhead. Consider a warm-up run before timing.

2. **Batch similar sequences**: Compilation is most effective when processing
   sequences of similar length in succession.

3. **Fixed conformations count**: Keeping the number of conformations constant
   across runs improves cache hits.

4. **GPU utilization**: Compilation benefits are most pronounced on GPUs where
   kernel fusion and memory access optimization provide significant gains.

5. **Profile first**: Use PyTorch profiling tools to identify bottlenecks
   before enabling compilation:

   .. code-block:: python

      import torch.profiler

      with torch.profiler.profile() as prof:
          ensemble = starling.generate(sequence, conformations=200)
      
      print(prof.key_averages().table(sort_by="cuda_time_total"))

Benchmarking Example
--------------------

Compare performance with and without compilation:

.. code-block:: python

   import time
   import starling

   sequence = "MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEAEKWPFFQEAQKLQAMHREKYPNYKYRPRRKAKMLPK"
   
   # Baseline: eager mode
   starling.set_compilation_options(enabled=False)
   start = time.time()
   for _ in range(10):
       ensemble = starling.generate(sequence, conformations=100)
   eager_time = time.time() - start
   
   # Compiled mode
   starling.set_compilation_options(enabled=True, mode="reduce-overhead")
   start = time.time()
   for _ in range(10):
       ensemble = starling.generate(sequence, conformations=100)
   compiled_time = time.time() - start
   
   print(f"Eager mode: {eager_time:.2f}s")
   print(f"Compiled mode: {compiled_time:.2f}s")
   print(f"Speedup: {eager_time/compiled_time:.2f}x")

Troubleshooting
---------------

Compilation Failures
^^^^^^^^^^^^^^^^^^^^

If you encounter compilation errors:

1. Try disabling ``fullgraph``:

   .. code-block:: python

      starling.set_compilation_options(
          enabled=True,
          fullgraph=False
      )

2. Use ``"default"`` mode instead of ``"reduce-overhead"``

3. Check PyTorch version – compilation support improves in newer releases

Slower Than Expected
^^^^^^^^^^^^^^^^^^^^

If compilation doesn't improve performance:

- Ensure you're running multiple iterations (compilation overhead is paid once)
- Check that you're using a GPU (CPU compilation benefits are smaller)
- Verify tensor shapes are consistent across runs
- Profile to identify non-compiled bottlenecks

Memory Issues
^^^^^^^^^^^^^

Compilation can increase memory usage:

- Reduce batch size or conformations count
- Use ``mode="default"`` instead of ``"max-autotune"``
- Monitor GPU memory with ``nvidia-smi``

See Also
--------

* :doc:`ensemble_generation` – Core sampling workflows and options
* :doc:`constraints` – Physics-based guidance during sampling
* :func:`starling.set_compilation_options` – API reference
* `PyTorch Compilation Documentation <https://pytorch.org/docs/stable/torch.compiler.html>`_
