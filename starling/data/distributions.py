import numpy as np
import torch


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        """
        Diagonal Gaussian distribution that can be used for sampling from given certain parameters.

        Parameters
        ----------
        parameters : torch.Tensor
            Parameters of the distribution.
        deterministic : bool, optional
            Whether to sample deterministically, by default False
        """

        self.parameters = parameters
        # Split the parameters into mean and logvar
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)

        # Clamp the logvar to prevent numerical instability
        #! Maybe switch to (-10, 10) - might help with numerical stability
        #! or self.std = torch.sqrt(torch.nn.functional.softplus(self.logvar))
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic

        # Compute the standard deviation and variance
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        # If deterministic sampling, set the variance to zero
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self) -> torch.Tensor:
        """
        Sample from the parameterized distribution.

        Returns
        -------
        torch.Tensor
            The sampled value.
        """
        x = self.mean + self.std * torch.randn(self.mean.shape, device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        """
        Return the mode of the distribution.

        Returns
        -------
        torch.Tensor
            Mode of the distribution.
        """
        return self.mean
