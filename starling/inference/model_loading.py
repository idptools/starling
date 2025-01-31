import os

# local imports
from starling.configs import DEFAULT_ENCODER_WEIGHTS_PATH, DEFAULT_DDPM_WEIGHTS_PATH
from starling.models.diffusion import DiffusionModel
from starling.models.unet import UNetConditional
from starling.models.vae import VAE
from starling import configs


class ModelManager:
    def __init__(self):
        self.encoder_model = None
        self.diffusion_model = None

    def load_models(self, encoder_path, ddpm_path, device):
        ''' Load the models if they are not already loaded. '''
        # check encoder model exists
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder model {encoder_path} not found.")

        # check ddpm model exists
        if not os.path.exists(ddpm_path):
            raise FileNotFoundError(f"DDPM model {ddpm_path} not found.")
        
        # Load the encoder model
        encoder_model = VAE.load_from_checkpoint(
            encoder_path,
            map_location=device
        )

        # Load the diffusion model
        diffusion_model = DiffusionModel.load_from_checkpoint(
            ddpm_path,
            model=UNetConditional(in_channels=1, out_channels=1, base=64,
                                  norm="group", blocks=[2, 2, 2], middle_blocks=2, labels_dim=configs.UNET_LABELS_DIM),
            encoder_model=encoder_model,
            map_location=device
        )

        return encoder_model, diffusion_model

    def get_models(self, encoder_path=DEFAULT_ENCODER_WEIGHTS_PATH, 
        ddpm_path=DEFAULT_DDPM_WEIGHTS_PATH, device='cpu'):
        '''
        Lazy-load models if not already loaded.

        Parameters
        ----------
        encoder_path : str
            The path to the encoder model.
            Default is 
        ddpm_path : str
            The path to the DDPM model.
        device : str
            The device on which to load the models. Default is CPU,
            but this changes depending on whatever we want to use 
            in ensemble_generation.py. Just made CPU default because
            all platforms have CPU. 

        Returns
        -------
        encoder_model, diffusion_model
            The loaded encoder and diffusion models.
        '''
        if self.encoder_model is None or self.diffusion_model is None:
            # Models haven't been loaded yet, so load them now
            self.encoder_model, self.diffusion_model = self.load_models(encoder_path, ddpm_path, device)

        # Return the already-loaded models
        return self.encoder_model, self.diffusion_model

