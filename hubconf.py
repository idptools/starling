dependencies = ["torch", "pytorch_lightning"]

from starling.inference.model_loading import ModelManager


def starling_model(pretrained=True, device="cpu"):
    """
    Entrypoint to load Starling models from PyTorch Hub

    Args:
        pretrained (bool): Must be True; STARLING only provides its
            pre-trained models through PyTorch Hub.
        device (str): Device to load the model on ('cpu' or 'cuda')

    Returns:
        tuple: (encoder_model, diffusion_model)

    Raises:
        NotImplementedError: If pretrained is False.
    """
    if not pretrained:
        raise NotImplementedError(
            "STARLING only provides pre-trained models through PyTorch Hub; pretrained=False is not supported."
        )
    model_manager = ModelManager()
    return model_manager.get_models(device=device)
