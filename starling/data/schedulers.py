import torch


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """
    The beta values are linearly spaced between 0.0001 and 0.02.

    Parameters
    ----------
    timesteps : int
        The number of timesteps which will be used to generate the beta values.

    Returns
    -------
    torch.Tensor
        A tensor containing the beta values.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    The beta values are generated using a cosine function. The beta values are
    clipped between 0.0001 and 0.9999.

    Parameters
    ----------
    timesteps : int
        The number of timesteps which will be used to generate the beta values.
    s : float, optional
        Adjusts the smoothness of the beta schedule's initial portion, influencing
        how quickly the values change over the first few timesteps. By default 0.008.

    Returns
    -------
    torch.Tensor
        A tensor containing the beta values.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sigmoid_beta_schedule(
    timesteps: int, start: int = 3, end: int = 3, tau: int = 1
) -> torch.Tensor:
    """
    The beta values are generated using a sigmoid function. The beta values are
    clipped between 0 and 0.999.

    Parameters
    ----------
    timesteps : int
        The number of timesteps which will be used to generate the beta values.
    start : int, optional
        The starting value for the sigmoid function, by default 3
    end : int, optional
        The ending value for the sigmoid function, by default 3
    tau : int, optional
        The time constant for the sigmoid function, by default 1

    Returns
    -------
    torch.Tensor
        A tensor containing the beta values.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
