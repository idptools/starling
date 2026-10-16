"""Construction of intrinsically disordered proteins ensembles through multiscale generative models"""

# Add imports here
import starling.configs
from typing import cast

# Import submodules to make them accessible as part of the top-level package
from starling.data import *
from starling.frontend.ensemble_generation import generate, sequence_encoder
from starling.models import *
from starling.structure.ensemble import load_ensemble
from starling.training import *

from ._version import __version__


def set_compilation_options(enabled=None, **torch_compile_kwargs):
    """
    Configure model compilation settings programmatically with full support for
    PyTorch compile parameters.

    Parameters
    ----------
    enabled : bool or None
        Whether to enable model compilation. If None, keeps current setting.
    **torch_compile_kwargs : keyword arguments
        Any valid arguments for ``torch.compile``, such as:

        - mode (str): Compilation mode ("default", "reduce-overhead", "max-autotune")
        - backend (str): Compilation backend ("inductor", "eager", "aot_eager", etc.)
        - fullgraph (bool): Whether to compile the full graph
        - dynamic (bool): Whether to handle dynamic shapes
        - disable (bool): Temporarily disable compilation
        - options (dict): Backend-specific options like {"triton.cudagraphs": True}

        ``torch.compile`` accepts ``mode`` or ``options`` but not both, so
        setting one removes the other from the stored settings (including
        the default ``mode="default"``).

    Raises
    ------
    ValueError
        If both ``mode`` and ``options`` are passed.

    Examples
    --------
    >>> import starling
    >>> # Basic usage
    >>> starling.set_compilation_options(enabled=True, mode="reduce-overhead")
    >>>
    >>> # Advanced usage with torch.compile parameters
    >>> starling.set_compilation_options(
    ...     enabled=True,
    ...     backend="inductor",
    ...     fullgraph=False,
    ...     dynamic=True,
    ...     options={"triton.cudagraphs": True}
    ... )
    >>> results = starling.generate(...)  # Uses compiled models with custom settings

    Returns
    -------
    dict
        Current compilation settings
    """
    from starling import configs
    from starling.inference.generation import model_manager

    # torch.compile raises if given both mode and options
    if "mode" in torch_compile_kwargs and "options" in torch_compile_kwargs:
        raise ValueError("torch.compile accepts either mode or options, not both")

    # Only update if values are provided
    if enabled is not None:
        configs.TORCH_COMPILATION["enabled"] = enabled

    # Update any provided options; whichever of mode/options is set last
    # replaces the other, since the stored defaults include mode="default"
    stored_options = cast(dict[str, object], configs.TORCH_COMPILATION["options"])
    if "options" in torch_compile_kwargs:
        stored_options.pop("mode", None)
    if "mode" in torch_compile_kwargs:
        stored_options.pop("options", None)
    for key, value in torch_compile_kwargs.items():
        stored_options[key] = value

    model_manager.clear()

    # Return current settings
    return {
        "enabled": configs.TORCH_COMPILATION["enabled"],
        "options": configs.TORCH_COMPILATION["options"].copy(),
    }
