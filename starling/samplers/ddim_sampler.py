from typing import List, Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from starling.data.data_wrangler import one_hot_encode


class DDIMSampler(nn.Module):
    def __init__(
        self,
        ddpm_model,
        n_steps: int,
        ddim_discretize: str = "uniform",
        ddim_eta: float = 0.0,
    ):
        super(DDIMSampler, self).__init__()
        self.ddpm_model = ddpm_model
        self.n_steps = 1000  # self.ddpm_model.n_steps
        self.ddim_discretize = ddim_discretize
        self.ddim_eta = ddim_eta

        if ddim_discretize == "uniform":
            c = self.n_steps // n_steps
            self.ddim_time_steps = np.asarray(list(range(0, self.n_steps, c))) + 1
        elif ddim_discretize == "quad":
            self.ddim_time_steps = (
                (np.linspace(0, np.sqrt(self.n_steps * 0.8), n_steps)) ** 2
            ).astype(int) + 1
        else:
            raise NotImplementedError(ddim_discretize)

        with torch.no_grad():
            alpha_bar = self.ddpm_model.alphas_cumprod
            self.ddim_alpha = alpha_bar[self.ddim_time_steps].clone().to(torch.float32)
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            self.ddim_alpha_prev = torch.cat(
                [alpha_bar[0:1], alpha_bar[self.ddim_time_steps[:-1]]]
            )
            self.ddim_sigma = (
                ddim_eta
                * (
                    (1 - self.ddim_alpha_prev)
                    / (1 - self.ddim_alpha)
                    * (1 - self.ddim_alpha / self.ddim_alpha_prev)
                )
                ** 0.5
            )

            self.ddim_sqrt_one_minus_alpha = (1.0 - self.ddim_alpha) ** 0.5

    def generate_labels(self, labels):
        labels = (
            torch.argmax(
                torch.from_numpy(one_hot_encode(labels.ljust(384, "0"))), dim=-1
            )
            .to(torch.int64)
            .squeeze()
            .to(self.ddpm_model.device)
        )
        labels = self.ddpm_model.sequence2labels(labels)

        return labels

    @torch.no_grad()
    def sample(
        self,
        shape: List[int],
        labels: torch.Tensor,
        repeat_noise: bool = False,
        temperature: float = 1.0,
        x_last: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.0,
        uncond_cond: Optional[torch.Tensor] = None,
        skip_steps: int = 0,
    ):
        device = self.ddpm_model.device
        bs = shape[0]

        x = torch.randn(shape, device=device)

        time_steps = np.flip(self.ddim_time_steps)

        labels = self.generate_labels(labels)

        for i, step in tqdm(enumerate(time_steps)):
            index = len(time_steps) - i - 1

            ts = x.new_full((bs,), step, dtype=torch.long)

            x, pred_x0, e_t = self.p_sample(
                x,
                labels,
                ts,
                step,
                index=index,
                repeat_noise=repeat_noise,
                temperature=temperature,
                uncond_scale=uncond_scale,
                uncond_cond=uncond_cond,
            )

        # Scale the latents back to the original scale
        x = (1 / self.ddpm_model.latent_space_scaling_factor) * x

        x = self.ddpm_model.encoder_model.decode(x)

        return x

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        step: int,
        index: int,
        *,
        repeat_noise: bool = False,
        temperature: float = 1.0,
        uncond_scale: float = 1.0,
        uncond_cond: Optional[torch.Tensor] = None,
    ):
        noise = self.ddpm_model.model(x, t, c)

        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(
            noise, index, x, temperature=temperature, repeat_noise=repeat_noise
        )

        return x_prev, pred_x0, noise

    def get_x_prev_and_pred_x0(
        self,
        e_t: torch.Tensor,
        index: int,
        x: torch.Tensor,
        *,
        temperature: float,
        repeat_noise: bool,
    ):
        alpha = self.ddim_alpha[index]

        alpha_prev = self.ddim_alpha_prev[index]

        sigma = self.ddim_sigma[index]

        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]

        pred_x0 = (x - sqrt_one_minus_alpha * e_t) / (alpha**0.5)

        dir_xt = (1.0 - alpha_prev - sigma**2).sqrt() * e_t

        if sigma == 0.0:
            noise = 0.0

        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
        else:
            noise = torch.randn(x.shape, device=x.device)

        noise = noise * temperature

        x_prev = (alpha_prev**0.5) * pred_x0 + dir_xt + sigma * noise

        return x_prev, pred_x0
