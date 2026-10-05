"""
STARLING's samplers.

Every sampler derives from BaseSampler (base_sampler.py) and can be built the
same way, ``SAMPLERS[name](ddpm_model, encoder_model, n_steps, ionic_strength)``;
helpers shared between samplers live in sampler_utilities.py.
"""

from __future__ import annotations

from typing import Final

from starling.samplers.base_sampler import BaseSampler
from starling.samplers.ddim_sampler import DDIMSampler
from starling.samplers.ddpm_sampler import DDPMSampler
from starling.samplers.dpmpp_sampler import DPMppSampler
from starling.samplers.plms_sampler import PLMSSampler

# the sampler class for each name generate() and the CLI accept
SAMPLERS: Final[dict[str, type[BaseSampler]]] = {
    "ddim": DDIMSampler,
    "ddpm": DDPMSampler,
    "dpmpp": DPMppSampler,
    "plms": PLMSSampler,
}

__all__ = [
    "SAMPLERS",
    "BaseSampler",
    "DDIMSampler",
    "DDPMSampler",
    "DPMppSampler",
    "PLMSSampler",
]
