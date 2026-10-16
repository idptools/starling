from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

import torch

from starling import configs

# local imports
from starling.configs import DEFAULT_DDPM_WEIGHTS_PATH, DEFAULT_ENCODER_WEIGHTS_PATH

if TYPE_CHECKING:
    from starling.models.diffusion import DiffusionModel
    from starling.models.vae import VAE


def weights_references(
    encoder_path: str | os.PathLike[str] | None,
    ddpm_path: str | os.PathLike[str] | None,
) -> tuple[str, str]:
    """
    Return the encoder and DDPM weight references ModelManager loads.

    ModelManager falls back on the default weight references that were in
    ``starling.configs`` when this module was imported, so this (not the
    current value of ``configs.DEFAULT_*_WEIGHTS_PATH``) is what should be
    recorded as the weights an ensemble was generated with.

    Parameters
    ----------
    encoder_path : str, os.PathLike or None
        Custom encoder checkpoint path or URL, or None for the default.
    ddpm_path : str, os.PathLike or None
        Custom diffusion model checkpoint path or URL, or None for the default.

    Returns
    -------
    tuple of (str, str)
        The encoder and DDPM weight references (local paths or URLs) that
        get_models() loads for these arguments.
    """
    return (
        os.fspath(encoder_path or DEFAULT_ENCODER_WEIGHTS_PATH),
        os.fspath(ddpm_path or DEFAULT_DDPM_WEIGHTS_PATH),
    )


class ModelManager:
    def __init__(self):
        self.encoder_model: VAE | None = None
        self.diffusion_model: DiffusionModel | None = None
        self._cache_key: tuple[str, str, str, str] | None = None

    def clear(self) -> None:
        self.encoder_model = None
        self.diffusion_model = None
        self._cache_key = None

    def load_models(
        self,
        encoder_path: str | os.PathLike[str] | None,
        ddpm_path: str | os.PathLike[str] | None,
        device: str | torch.device,
    ) -> tuple[VAE, DiffusionModel]:
        """Load the models from local files or URLs."""

        # Resolve paths: local files (~/.starling_weights, torch hub cache)
        # are preferred; downloads only happen if STARLING_OFFLINE is unset.
        encoder_path = configs.resolve_weights_path(encoder_path, DEFAULT_ENCODER_WEIGHTS_PATH)
        ddpm_path = configs.resolve_weights_path(ddpm_path, DEFAULT_DDPM_WEIGHTS_PATH)

        # Continue with existing loading logic
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder model {encoder_path} not found.")
        if not os.path.exists(ddpm_path):
            raise FileNotFoundError(f"DDPM model {ddpm_path} not found.")

        # Checkpoint construction needs Lightning; importing the API or CLI does not.
        from starling.models.diffusion import DiffusionModel
        from starling.models.transformer import SequenceEncoder
        from starling.models.vae import VAE
        from starling.models.vit import ViT

        # Load on the CPU and only then move to the device. With
        # map_location=device the whole Lightning checkpoint lands on the GPU
        # first, including about 1 GB of optimizer state inference never uses,
        # which can run a small or shared GPU out of memory while loading.
        sequence_encoder = SequenceEncoder(12, 512, 8)
        diffusion_model = DiffusionModel.load_from_checkpoint(
            ddpm_path,
            model=ViT(12, 512, 8, 512),
            sequence_encoder=sequence_encoder,
            distance_map_encoder=encoder_path,
            map_location="cpu",
        )
        encoder_model = VAE.load_from_checkpoint(
            encoder_path,
            map_location="cpu",
        )

        # The diffusion model keeps its own copy of the VAE so training can
        # encode distance maps on the fly. Inference decodes with
        # encoder_model instead, so drop the copy rather than carry it onto
        # the device.
        diffusion_model.distance_map_encoder = None

        diffusion_model.to(device).eval()
        encoder_model.to(device).eval()
        return encoder_model, diffusion_model

    def get_models(
        self,
        encoder_path: str | os.PathLike[str] | None = DEFAULT_ENCODER_WEIGHTS_PATH,
        ddpm_path: str | os.PathLike[str] | None = DEFAULT_DDPM_WEIGHTS_PATH,
        device: str | torch.device = "cpu",
    ) -> tuple[VAE, DiffusionModel]:
        """
        Lazy-load models if not already loaded.

        One model pair is cached. Changing checkpoint paths or device replaces it.

        Parameters
        ----------
        encoder_path : str, os.PathLike or None
            The path (or URL) to the encoder model. Default (and None) is
            configs.DEFAULT_ENCODER_WEIGHTS_PATH as it was when this module
            was imported.
        ddpm_path : str, os.PathLike or None
            The path (or URL) to the DDPM model. Default (and None) is
            configs.DEFAULT_DDPM_WEIGHTS_PATH as it was when this module
            was imported.
        device : str
            The device on which to load the models. Default is CPU,
            but this changes depending on whatever we want to use
            in ensemble_generation.py. Just made CPU default because
            all platforms have CPU.

        Returns
        -------
        encoder_model, diffusion_model
            The loaded encoder and diffusion models.
        """
        encoder_reference, ddpm_reference = weights_references(encoder_path, ddpm_path)
        key = (
            encoder_reference,
            ddpm_reference,
            str(torch.device(device)),
            repr((configs.TORCH_COMPILATION["enabled"], configs.TORCH_COMPILATION["options"])),
        )
        if self.encoder_model is None or self.diffusion_model is None or self._cache_key != key:
            # drop the old pair first, so replacing it never holds two pairs
            # of models in (GPU) memory at once
            self.clear()
            self.encoder_model, self.diffusion_model = self.load_models(encoder_reference, ddpm_reference, device)
            if configs.TORCH_COMPILATION["enabled"]:
                # Compile the models if requested
                self.encoder_model, self.diffusion_model = self.compile()
            self._cache_key = key

        # Return the already-loaded models
        return self.encoder_model, self.diffusion_model

    def compile(self) -> tuple[VAE, DiffusionModel]:
        """
        Compile the models using PyTorch's compile function.
        This is a placeholder for the actual compilation logic.
        """
        compile_kwargs = configs.TORCH_COMPILATION["options"].copy()

        if self.diffusion_model is None or self.encoder_model is None:
            raise RuntimeError("Models must be loaded before compiling")

        self.diffusion_model.model = cast(
            torch.nn.Module,
            torch.compile(self.diffusion_model.model, **compile_kwargs),
        )
        self.encoder_model.decoder = cast(
            torch.nn.Module,
            torch.compile(self.encoder_model.decoder, **compile_kwargs),
        )

        # self.diffusion_model.sequence_encoder = torch.compile(
        #     self.diffusion_model.sequence_encoder, **compile_kwargs
        # )

        print("\nCompiling the diffusion model for faster inference, this may take a while...")
        print("This is a one-time operation, subsequent inferences will be MUCH faster.\n")
        print("Compiling with the following options:")
        for key, value in compile_kwargs.items():
            print(f"  {key}: {value}")

        return self.encoder_model, self.diffusion_model
