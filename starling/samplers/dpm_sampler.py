import torch
from torch import nn
from tqdm import tqdm, trange

# Adapted from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


class DPMSolver(nn.Module):
    def __init__(
        self,
        ddpm_model,
        n_steps: int,
    ):
        super(DPMSolver, self).__init__()

        self.ddpm_model = ddpm_model
        self.n_steps = self.n_steps
        
        self.alpha_bar = self.ddpm_model.alphas_cumprod
        
        # Compute sigma_min and sigma_max
        self.sigma_min = torch.sqrt(1 - alpha_bar[0])  # At t=0
        self.sigma_max = torch.sqrt(1 - alpha_bar[-1])  # At t=T
        
        # Get the sigmas according to Karras et al. (2022) schedule
        self.sigmas = get_sigmas_karras(n_steps, sigma_min, sigma_max)
        
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

        labels = labels.unsqueeze(0)

        return labels

    @torch.no_grad()
    def sample(self,
        num_conformations: int,
        labels: torch.Tensor,
        show_per_step_progress_bar: bool = True):
        """DPM-Solver++(2M)."""
        
        device = self.ddpm_model.device

        # Initialize the latents with noise
        x = torch.randn(
            [
                num_conformations,
                self.ddpm_model.in_channels,
                self.ddpm_model.image_size,
                self.ddpm_model.image_size,
            ],
            device=device,
        ) * self.sigma_max
        
        labels = self.generate_labels(labels)
        
        s_in = x.new_ones([x.shape[0]])
        def sigma_fn(t):
            return t.neg().exp()
        
        def t_fn(sigma):
            return sigma.log().neg()
        
        old_denoised = None

        for i in trange(len(self.sigmas) - 1, disable=disable):
            
            # Map continuous sigma to discrete timestep
            sigma = self.sigmas[i]
            # Compute corresponding alpha_bar
            alpha_bar = 1 / (1 + sigma**2)
            # Nearest discrete timestep
            timestep = (torch.abs(self.alpha_bar - alpha_bar)).argmin().item()

            # Model prediction
            denoised = self.ddpm_model(x, timestep, labels)
            
            # Update the sample based on DPM++ 2M
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            if old_denoised is None or sigmas[i + 1] == 0:
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
            old_denoised = denoised
        return x
