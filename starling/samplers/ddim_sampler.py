from typing import List, Optional

import numpy as np
import torch
from torch import nn


class DDIMSampler(nn.Module):
    def __init__(
        self,
        model,
        n_steps: int,
        ddim_discretize: str = "uniform",
        ddim_eta: float = 0.0,
    ):
        super(DDIMSampler, self).__init__()
        self.model = model
        self.n_steps = n_steps
        self.ddim_discretize = ddim_discretize
        self.ddim_eta = ddim_eta

        # self.model_steps = self.model.timesteps

        if ddim_discretize == "uniform":
            c = self.n_steps // n_steps
            self.ddim_time_steps = np.asarray(list(range(0, self.n_steps, c))) + 1

        with torch.no_grad():
            alpha_bar = self.model.alphas_cumprod
            self.ddim_alpha = alpha_bar[self.time_steps].clone().to(torch.float32)
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            self.ddim_alpha_prev = torch.cat(
                [alpha_bar[0:1], alpha_bar[self.time_steps[:-1]]]
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

    @torch.no_grad()
    def sample(
        self,
        shape: List[int],
        cond: torch.Tensor,
        repeat_noise: bool = False,
        temperature: float = 1.0,
        x_last: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.0,
        uncond_cond: Optional[torch.Tensor] = None,
        skip_steps: int = 0,
    ):
        device = self.model.device
        bs = shape[0]

        x = torch.randn(shape, device=device)

        time_steps = np.flip(self.time_steps)

        for i, step in enumerate(time_steps):
            index = len(time_steps) - i - 1

            ts = x.new_full((bs,), step, dtype=torch.long)

            x, pred_x0, e_t = self.p_sample(
                x,
                cond,
                ts,
                step,
                index=index,
                repeat_noise=repeat_noise,
                temperature=temperature,
                uncond_scale=uncond_scale,
                uncond_cond=uncond_cond,
            )

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
        noise = self.model(x, t, c)

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
