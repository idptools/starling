from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from starling.data.tokenizer import StarlingTokenizer

if TYPE_CHECKING:
    from starling.models.diffusion import DiffusionModel
    from starling.models.vae import VAE


class DPMppSampler(nn.Module):
    """Second-order deterministic multistep solver with first-order endpoints.

    Timesteps are rounded uniform half-log-SNR points, ending at timestep 1.
    The upstream cosine-schedule cutoff
    (lambda >= -5.1) avoids unstable noise-to-data conversion near zero SNR.
    ``n_steps`` counts model evaluations; integer collisions reduce this count
    with a warning. Constraints are not supported.
    """

    def __init__(
        self,
        ddpm_model: DiffusionModel,
        encoder_model: VAE,
        n_steps: int,
        ionic_strength: float = 150,
    ) -> None:
        super().__init__()
        total = ddpm_model.num_timesteps
        if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or not 1 <= n_steps <= total:
            raise ValueError(f"n_steps must be an integer between 1 and {total}")
        self.ddpm_model = ddpm_model
        self.encoder_model = encoder_model
        self.device = ddpm_model.alphas_cumprod.device
        self.tokenizer = StarlingTokenizer()
        self.ionic_strength = torch.tensor([[ionic_strength]], device=self.device)
        abar = ddpm_model.alphas_cumprod.detach().to(device=self.device, dtype=torch.float32)
        self.alpha = abar.sqrt()
        self.sigma = (1 - abar).sqrt()
        self.lambda_t = self.alpha.log() - self.sigma.log()
        lam = self.lambda_t.cpu().numpy()
        usable = np.flatnonzero(np.isfinite(lam) & (lam >= -5.1))
        if usable.size < 2 or usable[-1] <= 1:
            raise ValueError("DPM++ requires a finite, decreasing log-SNR schedule")
        top = int(usable[-1])
        if not np.all(np.isfinite(lam[1 : top + 1])) or not np.all(np.diff(lam[1 : top + 1]) < 0):
            raise ValueError("DPM++ requires a finite, decreasing log-SNR schedule")
        targets = np.linspace(lam[top], lam[1], n_steps + 1)
        raw = np.rint(np.interp(targets, lam[1 : top + 1][::-1], np.arange(1, top + 1)[::-1])).astype(int)
        self.timesteps = np.unique(raw)[::-1].tolist()
        self.n_steps = len(self.timesteps) - 1
        if self.n_steps != n_steps:
            warnings.warn(
                f"DPM++ requested {n_steps} steps; using {self.n_steps} distinct steps at this model's timestep resolution",
                stacklevel=2,
            )

    @torch.no_grad()
    def sample(
        self,
        num_conformations: int,
        labels: str,
        show_per_step_progress_bar: bool = True,
        batch_count: int = 1,
        max_batch_count: int = 1,
        constraint: object = None,
    ) -> torch.Tensor:
        """Return decoded, uncropped maps of shape (num_conformations, 1, H, W)."""
        if constraint is not None:
            raise ValueError("DPM++ does not support constraints; use DDIM or DDPM")
        if (
            isinstance(num_conformations, bool)
            or not isinstance(num_conformations, (int, np.integer))
            or num_conformations < 1
        ):
            raise ValueError("num_conformations must be a positive integer")
        tokens = torch.tensor([self.tokenizer.encode(labels)], device=self.device)
        mask = torch.ones_like(tokens, dtype=torch.bool)
        context = self.ddpm_model.sequence2labels(tokens, mask, self.ionic_strength)
        x = torch.randn(num_conformations, 1, 24, 24, device=self.device)
        previous = previous_time = None
        pairs = zip(self.timesteps, self.timesteps[1:])
        for i, (source, target) in enumerate(
            tqdm(
                pairs,
                total=self.n_steps,
                disable=not show_per_step_progress_bar,
                position=1,
                leave=False,
                desc=f"DPM++ steps (batch {batch_count} of {max_batch_count})",
            )
        ):
            ts = torch.full((num_conformations,), source, device=self.device, dtype=torch.long)
            noise = self.ddpm_model.model(x, ts, context, mask).float()
            clean = (x - self.sigma[source] * noise) / self.alpha[source]
            h = self.lambda_t[target] - self.lambda_t[source]
            estimate = clean
            if previous is not None and i != self.n_steps - 1:
                ratio = (self.lambda_t[source] - self.lambda_t[previous_time]) / h
                estimate = clean + (clean - previous) / (2 * ratio)
            x = self.sigma[target] / self.sigma[source] * x - self.alpha[target] * torch.expm1(-h) * estimate
            previous, previous_time = clean, source
        return self.encoder_model.decode(x / self.ddpm_model.latent_space_scaling_factor)
