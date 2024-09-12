from typing import Tuple

import numpy as np
import torch
from IPython import embed
from torch import nn
from tqdm import tqdm

from starling.data.data_wrangler import one_hot_encode


class PNDMSampler(nn.Module):
    def __init__(
        self,
        ddpm_model,
        n_steps: int,
    ):
        super(PNDMSampler, self).__init__()

        self.ddpm_model = ddpm_model
        self.n_steps = n_steps
        self.total_steps = 1000

        self.ddpm_alphas_cumprod = self.ddpm_model.alphas_cumprod
        self.ets = []

    def __transfer(self, x, t, t_next, et, alphas_cump):
        at = alphas_cump[t.long() + 1].view(-1, 1, 1, 1)
        at_next = alphas_cump[t_next.long() + 1].view(-1, 1, 1, 1)

        x_delta = (at_next - at) * (
            (1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x
            - 1
            / (at.sqrt() * (((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt()))
            * et
        )

        x_next = x + x_delta
        return x_next

    def __runge_kutta(self, x, t_list, model, alphas_cump, ets, labels):
        e_1 = model(x, t_list[0], labels)
        ets.append(e_1)
        x_2 = self.__transfer(x, t_list[0], t_list[1], e_1, alphas_cump)

        e_2 = model(x_2, t_list[1], labels)
        x_3 = self.__transfer(x, t_list[0], t_list[1], e_2, alphas_cump)

        e_3 = model(x_3, t_list[1], labels)
        x_4 = self.__transfer(x, t_list[0], t_list[2], e_3, alphas_cump)

        e_4 = model(x_4, t_list[2], labels)
        et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)

        return et

    def gen_order_4(self, img, t, t_next, model, alphas_cump, ets, labels):
        t_list = [t, (t + t_next) / 2, t_next]
        if len(ets) > 2:
            noise_ = model(img, t, labels)
            ets.append(noise_)
            noise = (1 / 24) * (
                55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]
            )
        else:
            noise = self.__runge_kutta(img, t_list, model, alphas_cump, ets, labels)

        img_next = self.__transfer(img, t, t_next, noise, alphas_cump)

        return img_next

    def generate_labels(self, labels: str) -> torch.Tensor:
        """
        Generate labels to condition the generative process on.

        Parameters
        ----------
        labels : str
            A sequence to generate labels from.

        Returns
        -------
        torch.Tensor
            The labels to condition the generative process on.
        """
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

    def sample(
        self,
        num_conformations: int,
        labels: torch.Tensor,
    ):
        device = self.ddpm_model.device

        latent = torch.randn(
            [
                num_conformations,
                self.ddpm_model.in_channels,
                self.ddpm_model.image_size,
                self.ddpm_model.image_size,
            ],
            device=device,
        )
        labels = self.generate_labels(labels)

        skip = self.total_steps // self.n_steps
        steps = range(0, self.total_steps, skip)

        with torch.no_grad():
            # imgs = [noise]
            seq_next = [-1] + list(steps[:-1])

            for i, j in tqdm(zip(reversed(steps), reversed(seq_next))):
                t = (torch.ones(num_conformations) * i).to(device)
                t_next = (torch.ones(num_conformations) * j).to(device)

                # img_t = imgs[-1].to(device)

                latent = self.gen_order_4(
                    latent,
                    t,
                    t_next,
                    self.ddpm_model.model,
                    self.ddpm_alphas_cumprod,
                    self.ets,
                    labels,
                )

        latent = (1 / self.ddpm_model.latent_space_scaling_factor) * latent
        distance_map = self.ddpm_model.encoder_model.decode(latent)

        return distance_map
