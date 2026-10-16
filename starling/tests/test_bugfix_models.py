"""
Regression tests for fixes to STARLING's model and data modules.

Everything here runs on the CPU in a few seconds with tiny model instances, so
no checkpoints are needed.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Final

import pytest
import torch
from torch.nn import functional as F

from starling.data.positional_encodings import LearnablePositionalEncoding1D, LearnablePositionalEncoding2D
from starling.data.schedulers import cosine_beta_schedule, linear_beta_schedule, sigmoid_beta_schedule
from starling.models import diffusion as diffusion_module
from starling.models import vae as vae_module
from starling.models.attention import MultiHeadAttention
from starling.models.diffusion import DiffusionBatch, DiffusionModel
from starling.models.normalization import RMSNorm
from starling.models.transformer import DIT_LAYER_NORM_EPS, DiTBlock, SequenceEncoder
from starling.models.vae import VAE, KLDWeightScheduler

# a tiny sequence encoder: 2 layers, 16-dimensional embeddings, 2 heads
TINY_LAYERS: Final[int] = 2
TINY_EMBED_DIM: Final[int] = 16
TINY_HEADS: Final[int] = 2

# ionic strength the sequences are conditioned on, in mM
IONIC_STRENGTH_MM: Final[float] = 150.0

# float32 outputs of the same computation on differently padded batches agree
# to well within this
FLOAT32_TOLERANCE: Final[float] = 1e-5

# a training run of 50 epochs of 100 steps (fewer than the 100 epochs below
# which the diffusion model's warmup used to round down to zero steps)
EPOCHS: Final[int] = 50
STEPS_PER_EPOCH: Final[int] = 100


def plain_softmax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention as a plain masked softmax.

    This is how torch < 2.5 computed scaled_dot_product_attention: a row whose
    keys are all masked out has softmax over nothing but -inf, which is NaN.

    Parameters
    ----------
    query : torch.Tensor
        Queries, shape (batch, heads, N, head_dim).
    key : torch.Tensor
        Keys, shape (batch, heads, S, head_dim).
    value : torch.Tensor
        Values, shape (batch, heads, S, head_dim).
    attn_mask : torch.Tensor or None, optional
        Boolean mask, True where a query may attend to a key. Default None.

    Returns
    -------
    torch.Tensor
        Attention output, shape (batch, heads, N, head_dim).
    """
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ value


def encode_padded_and_unpadded(encoder: SequenceEncoder) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a 5-residue sequence on its own and padded to 8 residues in a batch.

    Parameters
    ----------
    encoder : SequenceEncoder
        The sequence encoder, in eval mode.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor)
        The encoding of the sequence on its own, shape (1, 5, embed_dim), and
        the encoding of the whole padded batch, shape (2, 8, embed_dim).
    """
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    padded_tokens = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0], [6, 7, 8, 9, 10, 11, 12, 13]])
    ionic_strength = torch.tensor([[IONIC_STRENGTH_MM]])
    with torch.no_grad():
        unpadded = encoder(tokens, tokens != 0, ionic_strength)
        padded = encoder(padded_tokens, padded_tokens != 0, ionic_strength.repeat(2, 1))
    return unpadded, padded


# ---------------------------------------------------------------- transformer


def test_dit_layer_norms_keep_the_eps_the_released_weights_were_trained_with():
    """Every DiT LayerNorm runs with eps = 512 whatever the context dimension."""
    assert DIT_LAYER_NORM_EPS == 512
    block = DiTBlock(TINY_EMBED_DIM, TINY_HEADS, context_dim=24)
    assert [block.norm1.eps, block.norm2.eps, block.norm3.eps] == [DIT_LAYER_NORM_EPS] * 3


def test_padding_does_not_change_the_encoding_of_real_residues():
    """Real residues are encoded the same whether or not the batch is padded."""
    torch.manual_seed(0)
    encoder = SequenceEncoder(TINY_LAYERS, TINY_EMBED_DIM, TINY_HEADS).eval()
    unpadded, padded = encode_padded_and_unpadded(encoder)
    torch.testing.assert_close(padded[:1, :5], unpadded, rtol=0, atol=FLOAT32_TOLERANCE)


def test_padded_queries_give_no_nan_with_plain_softmax_attention(monkeypatch):
    """With torch < 2.5's attention, fully masked padded rows gave NaN that spread to real residues."""
    torch.manual_seed(0)
    encoder = SequenceEncoder(TINY_LAYERS, TINY_EMBED_DIM, TINY_HEADS).eval()
    monkeypatch.setattr(F, "scaled_dot_product_attention", plain_softmax_attention)

    unpadded, padded = encode_padded_and_unpadded(encoder)

    assert torch.isfinite(padded).all()
    torch.testing.assert_close(padded[:1, :5], unpadded, rtol=0, atol=FLOAT32_TOLERANCE)


def test_padded_queries_leave_attention_as_the_output_bias():
    """Padded queries get a zero attention output, i.e. out_proj's bias, as torch >= 2.5 gave."""
    torch.manual_seed(0)
    attention = MultiHeadAttention(TINY_EMBED_DIM, TINY_HEADS).eval()
    tokens = torch.randn(1, 4, TINY_EMBED_DIM)
    mask = torch.tensor([[True, True, False, False]])
    with torch.no_grad():
        output = attention(tokens, tokens, query_mask=mask, context_mask=mask)
    torch.testing.assert_close(output[0, 2:], attention.out_proj.bias.expand(2, -1), rtol=0, atol=0)


# ---------------------------------------------------------------- schedulers


@pytest.mark.parametrize("schedule", [linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule])
def test_beta_schedules_are_float32(schedule):
    """Every schedule gives float32 betas, so the diffusion model's buffers are float32."""
    assert schedule(1000).dtype == torch.float32


@pytest.mark.parametrize("beta_scheduler", ["linear", "cosine", "sigmoid"])
def test_noising_stays_float32_with_every_schedule(beta_scheduler):
    """q_sample used to promote latents to float64 with the linear and sigmoid schedules."""
    model = DiffusionModel(torch.nn.Identity(), torch.nn.Identity(), None, beta_scheduler=beta_scheduler)
    assert model.sqrt_alphas_cumprod.dtype == torch.float32
    latents = torch.randn(2, 1, 24, 24)
    noised = model.q_sample(latents, torch.tensor([3, 700]))
    assert noised.dtype == torch.float32


# ---------------------------------------------------------------- learning-rate warmup


def warmup_multipliers(module: DiffusionModel | VAE, steps: list[int]) -> list[float]:
    """
    Learning-rate multipliers of LinearWarmupCosineAnnealingLR at some steps.

    Parameters
    ----------
    module : DiffusionModel or VAE
        A model whose config_scheduler is LinearWarmupCosineAnnealingLR.
    steps : list of int
        Training steps to evaluate the multiplier at.

    Returns
    -------
    list of float
        The multiplier of the initial learning rate at each step.
    """
    # stand in for the Lightning trainer, which only has to report the run's length
    setattr(module, "_trainer", SimpleNamespace(max_epochs=EPOCHS, estimated_stepping_batches=EPOCHS * STEPS_PER_EPOCH))
    _, [scheduler] = module.configure_optimizers()
    lr_lambda = scheduler["scheduler"].lr_lambdas[0]
    return [lr_lambda(step) for step in steps]


def tiny_vae(**kwargs: object) -> VAE:
    """
    A tiny VAE for 32x32 maps, built without compilation.

    Parameters
    ----------
    **kwargs : object
        Arguments that replace the defaults below.

    Returns
    -------
    VAE
        The VAE.
    """
    arguments: dict = dict(
        model_type="Resnet18",
        in_channels=1,
        latent_dim=1,
        dimension=32,
        loss_type="mse",
        KLD_weight=1e-4,
        lr_scheduler="LinearWarmupCosineAnnealingLR",
        set_lr=1e-3,
        norm="instance",
        base=8,
        optimizer="Adam",
        compile_mode=None,
    )
    arguments.update(kwargs)
    return VAE(**arguments)


@pytest.mark.parametrize("module_name", ["diffusion", "vae"])
def test_learning_rate_warms_up_over_one_percent_of_training(module_name):
    """Warmup lasts WARMUP_FRACTION of all steps; the diffusion model's used to be zero below 100 epochs."""
    if module_name == "diffusion":
        module = DiffusionModel(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1), None)
        warmup_fraction = diffusion_module.WARMUP_FRACTION
    else:
        module = tiny_vae()
        warmup_fraction = vae_module.WARMUP_FRACTION
    assert warmup_fraction == 0.01

    warmup_steps = int(EPOCHS * STEPS_PER_EPOCH * warmup_fraction)
    start, halfway, end = warmup_multipliers(module, [0, warmup_steps // 2, warmup_steps])
    assert start == 0.0
    assert halfway == pytest.approx(0.5)
    assert end == pytest.approx(1.0)


# ---------------------------------------------------------------- KLD weight


def test_linear_kld_schedule_without_warmup_uses_the_maximum_weight():
    """A warmup fraction of 0 used to divide by zero warmup steps."""
    scheduler = KLDWeightScheduler(max_weight=0.1, warmup_fraction=0, scheduler_type="linear")
    scheduler.configure(1000)
    assert scheduler.get_weight(5) == 0.1


def test_cyclical_kld_schedule_on_a_run_too_short_for_a_cycle_uses_the_maximum_weight():
    """Fewer than 5 training steps make a cycle 0 steps long, which used to be step % 0."""
    scheduler = KLDWeightScheduler(max_weight=0.1, warmup_fraction=0.5, scheduler_type="cyclical")
    scheduler.configure(4)
    assert scheduler.get_weight(3) == 0.1


def test_kld_schedules_still_ramp_up():
    """The linear and cyclical ramps are unchanged by the fixes."""
    linear = KLDWeightScheduler(max_weight=1.0, warmup_fraction=0.1, scheduler_type="linear")
    linear.configure(1000)
    assert [linear.get_weight(step) for step in (0, 50, 100, 500)] == [0.0, 0.5, 1.0, 1.0]

    # 1000 steps make cycles of 200 steps, the first 100 of which ramp up
    cyclical = KLDWeightScheduler(max_weight=1.0, warmup_fraction=0.5, scheduler_type="cyclical")
    cyclical.configure(1000)
    assert [cyclical.get_weight(step) for step in (0, 50, 150, 250)] == [0.0, 0.5, 1.0, 0.5]


def test_unknown_kld_schedule_is_rejected():
    """An unknown scheduler type used to fail later with an UnboundLocalError."""
    with pytest.raises(ValueError, match="scheduler_type"):
        KLDWeightScheduler(max_weight=0.1, warmup_fraction=0.1, scheduler_type="exponential")


# ---------------------------------------------------------------- VAE normalization


def test_vae_supports_rms_norm():
    """VAE(norm="rms") used to raise KeyError('rms') in the encoder's first layer."""
    vae = tiny_vae(norm="rms")
    assert any(isinstance(module, RMSNorm) for module in vae.encoder.modules())
    assert any(isinstance(module, RMSNorm) for module in vae.decoder.modules())
    reconstructed, _ = vae(torch.rand(2, 1, 32, 32))
    assert reconstructed.shape == (2, 1, 32, 32)


# ---------------------------------------------------------------- latent scaling


def diffusion_batch() -> DiffusionBatch:
    """
    A training batch of two latents with a standard deviation of about 2.

    Returns
    -------
    DiffusionBatch
        The batch, as DiffusionModel.training_step() expects it.
    """
    generator = torch.Generator().manual_seed(0)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 0]])
    return {
        "data": 2.0 * torch.randn(2, 1, 24, 24, generator=generator),
        "sequence": tokens,
        "attention_mask": tokens != 0,
        "ionic_strengths": torch.tensor([[IONIC_STRENGTH_MM], [IONIC_STRENGTH_MM]]),
    }


def run_first_training_step(model: DiffusionModel, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Run training_step() on the first batch, without the network or a trainer.

    Parameters
    ----------
    model : DiffusionModel
        The diffusion model. Its loss and its all_gather (a single process)
        are replaced, so only the latent scaling logic runs.
    monkeypatch : pytest.MonkeyPatch
        pytest's monkeypatch fixture, used to replace them.
    """
    monkeypatch.setattr(model, "forward", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(model, "all_gather", lambda value: value)
    with pytest.warns(UserWarning, match="self.log"):
        model.training_step(diffusion_batch(), batch_idx=0)


def test_first_training_step_sets_the_latent_scaling_factor(monkeypatch):
    """A fresh model scales its latents by 1 / std of the first batch."""
    model = DiffusionModel(torch.nn.Identity(), torch.nn.Identity(), None)
    run_first_training_step(model, monkeypatch)
    expected = 1.0 / diffusion_batch()["data"].std()
    torch.testing.assert_close(model.latent_space_scaling_factor, expected)


def test_fine_tuning_keeps_the_latent_scaling_factor_from_the_checkpoint(monkeypatch):
    """A factor loaded from a checkpoint used to be replaced by a one-batch estimate at step 0."""
    model = DiffusionModel(torch.nn.Identity(), torch.nn.Identity(), None)
    model.latent_space_scaling_factor.fill_(0.3)
    run_first_training_step(model, monkeypatch)
    assert model.latent_space_scaling_factor.item() == pytest.approx(0.3)


# ---------------------------------------------------------------- positional encodings


@pytest.mark.parametrize(
    "encoding",
    [LearnablePositionalEncoding1D(5, TINY_EMBED_DIM), LearnablePositionalEncoding2D(3, 3, TINY_EMBED_DIM)],
    ids=["1d", "2d"],
)
def test_learnable_positional_encodings_follow_the_module(encoding):
    """The encoding moves with .to() and stays a Parameter; it used to be reassigned a plain tensor."""
    encoding = encoding.to("meta")
    inputs = torch.empty(encoding.positional_encoding.shape, device="meta")
    assert encoding(inputs).device.type == "meta"
    assert isinstance(encoding.positional_encoding, torch.nn.Parameter)
    assert len(list(encoding.parameters())) == 1


def test_learnable_positional_encoding_on_another_device_fails_loudly():
    """An input on another device raises torch's device error rather than a TypeError."""
    encoding = LearnablePositionalEncoding1D(5, TINY_EMBED_DIM)
    with pytest.raises(RuntimeError, match="device"):
        encoding(torch.empty(1, 5, TINY_EMBED_DIM, device="meta"))
