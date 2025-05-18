import sys
from typing import Tuple

import numpy as np
import torch
from einops import rearrange
from torch import nn
from tqdm.auto import tqdm

from starling.data.tokenizer import StarlingTokenizer


class DDIMSampler(nn.Module):
    def __init__(
        self,
        ddpm_model,
        encoder_model,
        n_steps: int,
        ddim_discretize: str = "uniform",
        ddim_eta: float = 0.0,
    ):
        """
        An efficient sampler that generates samples 10x to 100x faster than the DDPM model.
        Denoising diffusion implicit models (DDIM) do not require sampling the entire diffusion process to generate samples.
        The forward process is parameterized using non-Markovian diffusion processes, leading to short generative Markov chains
        that can be simulated in fewer steps.

        References
        ----------
        [1] Ho, J., Jaini, P., Hariharan, B., Abbeel, P., & Duan, Y. (2020).
        Denoising diffusion implicit models. arXiv preprint arXiv:2012.02142.

        Parameters
        ----------
        ddpm_model : _type_
            The trained DDPM model.
        n_steps : int
            The number of steps to simulate the generative process, smaller than the number of steps used to train the DDPM model.
        ddim_discretize : str, optional
            The discretization method for the generative process, by default "uniform".
        ddim_eta : float, optional
            The noise level for the generative process, a number between 0.0 and 1.0.
            0.0 adds no noise to the generative process, 1.0 adds the maximum noise.
            This number interpolates between deterministic and stochastic generative processes,
            by default 0.0.

        Raises
        ------
        NotImplementedError
            If the discretization method is not implemented.
        """
        super(DDIMSampler, self).__init__()
        self.ddpm_model = ddpm_model
        self.encoder_model = encoder_model
        self.n_steps = self.ddpm_model.num_timesteps
        self.ddim_discretize = ddim_discretize
        self.ddim_eta = ddim_eta
        self.tokenizer = StarlingTokenizer()

        # Ways to discretize the generative process
        if ddim_discretize == "uniform":
            c = self.n_steps // n_steps
            self.ddim_time_steps = np.asarray(list(range(0, self.n_steps - 1, c))) + 1
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
        labels = torch.tensor(self.tokenizer.encode(labels)).to(self.ddpm_model.device)
        labels = rearrange(labels, "f -> 1 f")
        attention_mask = torch.ones_like(labels, device=self.ddpm_model.device).bool()

        labels = self.ddpm_model.sequence2labels(labels, attention_mask)

        return labels

    @torch.no_grad()
    def sample(
        self,
        num_conformations: int,
        labels: torch.Tensor,
        repeat_noise: bool = False,
        temperature: float = 1.0,
        show_per_step_progress_bar: bool = True,
        batch_count: int = 1,
        max_batch_count: int = 1,
    ) -> torch.Tensor:
        """
        Sample the generative process using the DDIM model.

        Parameters
        ----------
        num_conformations : int
            Number of conformations to generate.

        labels : torch.Tensor
            The labels to condition the generative process on.

        repeat_noise : bool, optional
            _description_, by default False

        temperature : float, optional
            _description_, by default 1.0

        show_per_step_progress_bar : bool, optional
            whether to show progress bar per step.

        batch_count : int, optional
            The batch count for the progress bar, by default 1

        max_batch_count : int, optional
            The maximum batch count for the progress bar, by default 1

        Returns
        -------
        torch.Tensor
            The generated distance maps.
        """
        device = self.ddpm_model.device

        # Initialize the latents with noise
        x = torch.randn(
            [num_conformations, self.ddpm_model.in_channels, 24, 24],
            device=device,
        )

        time_steps = np.flip(self.ddim_time_steps)

        # Get the labels to condition the generative process on
        labels = self.generate_labels(
            labels,
        )

        # initialize progress bar if we want to show it
        if show_per_step_progress_bar:
            pbar_inner = tqdm(
                total=len(time_steps),
                position=1,
                leave=False,
                desc=f"DDPM steps (batch {batch_count} of {max_batch_count})",
            )

        # Denoise the initial latent
        for i, step in enumerate(time_steps):
            index = len(time_steps) - i - 1

            # Batch the timesteps
            ts = x.new_full((num_conformations,), step, dtype=torch.long)

            # Sample the generative process
            x, *_ = self.p_sample(
                x,
                labels,
                ts,
                step,
                index=index,
                repeat_noise=repeat_noise,
                temperature=temperature,
            )
            # update progress bar if we are showing it
            if show_per_step_progress_bar:
                pbar_inner.update(1)

        # if we have progress bar, close after finishing the steps.
        if show_per_step_progress_bar:
            pbar_inner.close()

        # Scale the latents back to the original scale
        x = (1 / self.ddpm_model.latent_space_scaling_factor) * x

        # Decode the latents to get the distance maps
        x = self.encoder_model.decode(x)

        return x

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        step: int,
        index: int,
        repeat_noise: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take one step in the generative process.

        Parameters
        ----------
        x : torch.Tensor
            The tensor to remove noise from.
        c : torch.Tensor
            The labels to condition the generative process on.
        t : torch.Tensor
            The timestep to sample the generative process at.
        step : int

        index : int
            _description_
        repeat_noise : bool, optional
            _description_, by default False
        temperature : float, optional
            _description_, by default 1.0

        Returns
        -------
        _type_
            _description_
        """

        # Predict the amount of noise in the latent based on the timestep and labels

        # print(f"x shape: {x.shape}")
        # print(f"c shape: {c.shape}")

        attention_mask = torch.ones((c.shape[0], c.shape[1]), device=x.device)
        attention_mask = attention_mask.bool()

        predicted_noise = self.ddpm_model.unet_model(x, t, c, attention_mask)

        # Calculate the previous latent and the predicted latent
        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(
            predicted_noise,
            index,
            x,
            temperature=temperature,
            repeat_noise=repeat_noise,
        )

        return x_prev, pred_x0, predicted_noise

    def get_x_prev_and_pred_x0(
        self,
        predicted_noise: torch.Tensor,
        index: int,
        x: torch.Tensor,
        temperature: float,
        repeat_noise: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Remove the noise from the latent iteratively.

        Parameters
        ----------
        predicted_noise : torch.Tensor
            The noise predicted by the DDPM model.
        index : int
            The index of the timestep.
        x : torch.Tensor
            The latent to remove the noise from.
        temperature : float
            The temperature to use for the generative process
        repeat_noise : bool
            Whether to repeat the noise

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The previous latent.
        """
        alpha = self.ddim_alpha[index]

        alpha_prev = self.ddim_alpha_prev[index]

        sigma = self.ddim_sigma[index]

        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]

        # Predicted x_0
        pred_x0 = (x - sqrt_one_minus_alpha * predicted_noise) / (alpha**0.5)

        # Direction pointing to x_t
        dir_xt = (1.0 - alpha_prev - sigma**2).sqrt() * predicted_noise

        if sigma == 0.0:
            noise = 0.0

        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
        else:
            noise = torch.randn(x.shape, device=x.device)

        noise = noise * temperature

        x_prev = (alpha_prev**0.5) * pred_x0 + dir_xt + sigma * noise

        return x_prev, pred_x0
