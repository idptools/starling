dependencies = ["torch", "pytorch_lightning"]

from starling.inference.model_loading import ModelManager


def starling_model(pretrained=True, device="cpu"):
    """
    Entrypoint to load Starling models from PyTorch Hub

    Args:
        pretrained (bool): If True, returns models pre-trained on the default dataset
        device (str): Device to load the model on ('cpu' or 'cuda')

    Returns:
        tuple: (encoder_model, diffusion_model)
    """
    model_manager = ModelManager()
    return model_manager.get_models(device=device)
