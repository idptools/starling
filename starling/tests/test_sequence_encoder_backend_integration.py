import os

import pytest
import torch

from starling import configs
from starling.inference.generation import sequence_encoder_backend
from starling.inference.model_loading import ModelManager


@pytest.mark.slow
@pytest.mark.integration
def test_sequence_encoder_backend_real_models():
    """Optional end-to-end test loading real weights.

    Skipped unless BOTH:
      1. Environment variable STARLING_RUN_INTEGRATION == "1"
      2. Default encoder & diffusion checkpoint files exist locally.

    This keeps the default test suite fast/lightweight while allowing
    manual or CI jobs to exercise the real model stack.
    """

    if os.getenv("STARLING_RUN_INTEGRATION") != "1":
        pytest.skip("STARLING_RUN_INTEGRATION env var not set to 1")

    enc_path = configs.DEFAULT_ENCODER_WEIGHTS_PATH
    ddpm_path = configs.DEFAULT_DDPM_WEIGHTS_PATH

    if not (os.path.exists(enc_path) and os.path.exists(ddpm_path)):
        pytest.skip("Model weight files not found; skipping integration test")

    mm = ModelManager()
    # Use tiny set of short sequences to minimize load + inference time
    sequences = {
        "short": "ACDEFG",
        "tiny": "AC",
    }

    out = sequence_encoder_backend(
        sequence_dict=sequences,
        device="cpu",  # keep CPU to avoid GPU requirement in generic environments
        batch_size=2,
        salt=150,
        output_directory=None,
        model_manager=mm,
        encoder_path=enc_path,
        ddpm_path=ddpm_path,
    )

    assert set(out.keys()) == set(sequences.keys())

    for name, seq in sequences.items():
        emb = out[name]
        # Expect 2D tensor (L, D)
        assert emb.ndim == 2
        assert emb.shape[0] == len(seq)
        # Basic numeric sanity checks
        assert emb.dtype in (torch.float32, torch.float16, torch.bfloat16)
        # Ensure embeddings are not all zeros / constant
        variance = emb.var().item()
        assert variance > 0, f"Embedding for {name} appears degenerate (var=0)"
