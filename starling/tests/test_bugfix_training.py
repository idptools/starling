"""
Regression tests for the training data loaders, training scripts and VAE evaluation.

Everything here runs on CPU against tiny synthetic webdataset shards, HDF5 files and
stub models, so no training is run and no model weights are needed. The training
entry points are exercised with a fake Trainer and logger so we can check what they
pass to Lightning without launching a run.
"""

import importlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import webdataset as wds
from omegaconf import DictConfig, OmegaConf

import starling.inference
from starling.data import data_wrangler
from starling.data.ddpm_loader_tar import DDPMDataLoader
from starling.data.distributions import DiagonalGaussianDistribution
from starling.data.VAE_loader_tar import VAEdataloader
from starling.inference import evaluate_vae
from starling.training import diffusion_train, vae_train

# Synthetic dataset layout: N_SHARDS shards per split with SAMPLES_PER_SHARD samples
# each. Every sample's ionic strength is a unique id (shard * ID_STRIDE + index), so
# a batch tells us exactly which sample it came from.
N_SHARDS = 8
SAMPLES_PER_SHARD = 5
ID_STRIDE = 100
N_SAMPLES = N_SHARDS * SAMPLES_PER_SHARD

# Size of the toy distance maps / latents and the padded sequence length; the
# loaders do not check the size, so tiny arrays keep the shards small
MAP_SIZE = 6
SEQUENCE_LENGTH = 8


def _npz_bytes(array: np.ndarray) -> bytes:
    """
    Serialize an array the way the STARLING shard writers do.

    Parameters
    ----------
    array : np.ndarray
        Array to store under the "array" key.

    Returns
    -------
    bytes
        Contents of an .npz file.
    """
    buffer = io.BytesIO()
    np.savez_compressed(buffer, array=array)
    return buffer.getvalue()


def _write_shards(dataset_dir: Path) -> None:
    """
    Write train and validation webdataset shards with uniquely identifiable samples.

    Parameters
    ----------
    dataset_dir : Path
        Directory that will contain train/ and validation/ subdirectories.
    """
    for split in ("train", "validation"):
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True)
        for shard in range(N_SHARDS):
            with wds.TarWriter(str(split_dir / f"{split}_{shard:04d}.tar")) as sink:
                for index in range(SAMPLES_PER_SHARD):
                    sample_id = shard * ID_STRIDE + index
                    sequence = np.zeros(SEQUENCE_LENGTH, dtype=np.int64)
                    sequence[: 3 + index] = 1
                    distance_map = np.full((MAP_SIZE, MAP_SIZE), 3.8, dtype=np.float32)
                    np.fill_diagonal(distance_map, 0)
                    sink.write(
                        {
                            "__key__": f"s{shard}_i{index}_{sample_id}mM",
                            "distance_map.npz": _npz_bytes(distance_map),
                            "sequence.npz": _npz_bytes(sequence),
                        }
                    )


def _all_sample_ids() -> Counter:
    """
    Return every sample id in the synthetic dataset, each counted once.

    Returns
    -------
    Counter
        Mapping from sample id to 1.
    """
    return Counter(
        shard * ID_STRIDE + index
        for shard in range(N_SHARDS)
        for index in range(SAMPLES_PER_SHARD)
    )


def _loader_config(dataset_dir: Path, num_workers: int, **overrides) -> DictConfig:
    """
    Build a tar-loader config for the synthetic dataset.

    Parameters
    ----------
    dataset_dir : Path
        Directory written by _write_shards.
    num_workers : int
        Number of dataloader worker processes.
    **overrides
        Extra or replacement config entries.

    Returns
    -------
    DictConfig
        Config accepted by DDPMDataLoader and VAEdataloader.
    """
    config = dict(
        dataset=str(dataset_dir),
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=2,
        shuffle_buffer=4,
        train_size=N_SAMPLES,
        val_size=N_SAMPLES,
        apply_filter=False,
    )
    config.update(overrides)
    return OmegaConf.create(config)


def _validation_ids(data_module: DDPMDataLoader) -> Counter:
    """
    Run one validation pass and count how often each sample id was seen.

    Parameters
    ----------
    data_module : DDPMDataLoader
        Data module whose setup() has been called.

    Returns
    -------
    Counter
        Mapping from sample id to the number of times it appeared.
    """
    seen: Counter = Counter()
    for batch in data_module.val_dataloader():
        seen.update(batch["ionic_strengths"][:, 0].tolist())
    return seen


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory) -> Path:
    """Synthetic train/validation shards shared by the loader tests."""
    dataset_dir = tmp_path_factory.mktemp("shards")
    _write_shards(dataset_dir)
    return dataset_dir


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def test_ddpm_validation_reads_every_shard_once_with_two_workers(shard_dir):
    """Shards must be split by worker once, not twice (which dropped 3/4 of them)."""
    data_module = DDPMDataLoader(_loader_config(shard_dir, num_workers=2), effective_batch_size=1)
    data_module.setup()

    assert _validation_ids(data_module) == _all_sample_ids()


def test_ddpm_validation_splits_shards_across_ranks(shard_dir, monkeypatch):
    """Under DDP each rank reads a disjoint subset of shards that together cover all of them."""
    seen_per_rank = []
    for rank in range(2):
        # webdataset reads the rank and world size from these environment variables
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", "2")
        config = _loader_config(shard_dir, num_workers=0, val_size=N_SAMPLES // 2)
        data_module = DDPMDataLoader(config, effective_batch_size=1)
        data_module.setup()
        seen_per_rank.append(_validation_ids(data_module))

    assert set(seen_per_rank[0]).isdisjoint(seen_per_rank[1])
    assert seen_per_rank[0] + seen_per_rank[1] == _all_sample_ids()


@pytest.mark.parametrize("loader_class", [DDPMDataLoader, VAEdataloader])
def test_loaders_work_without_worker_processes(shard_dir, loader_class):
    """prefetch_factor must not be passed when num_workers is 0 (PyTorch raises)."""
    data_module = loader_class(_loader_config(shard_dir, num_workers=0), effective_batch_size=1)
    data_module.setup()

    assert len(list(data_module.val_dataloader())) == N_SAMPLES
    assert len(list(data_module.train_dataloader())) == N_SAMPLES


def test_ddpm_metadata_only_fills_the_missing_size(tmp_path):
    """An explicit train_size or val_size must not be overwritten by metadata.json."""
    metadata = {"a": {"train": 60, "validation": 20}, "b": {"train": 40, "validation": 10}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    effective_batch_size = 5

    only_train = _loader_config(tmp_path, num_workers=0, train_size=30, val_size=None)
    data_module = DDPMDataLoader(only_train, effective_batch_size=effective_batch_size)
    assert data_module.n_train_batches == 30 // effective_batch_size
    assert data_module.n_val_batches == 30 // effective_batch_size

    only_val = _loader_config(tmp_path, num_workers=0, train_size=None, val_size=15)
    data_module = DDPMDataLoader(only_val, effective_batch_size=effective_batch_size)
    assert data_module.n_train_batches == 100 // effective_batch_size
    assert data_module.n_val_batches == 15 // effective_batch_size


# ---------------------------------------------------------------------------
# Training scripts
# ---------------------------------------------------------------------------


class _FakeTrainer:
    """Stand-in for pl.Trainer that records its arguments instead of training."""

    # Pretend Lightning resolved the devices to 4 processes (e.g. devices=-1 on 4 GPUs)
    world_size = 4

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_ckpt_path = "fit was not called"

    def fit(self, model, datamodule, ckpt_path=None):
        self.fit_ckpt_path = ckpt_path


class _FakeLogger:
    """Stand-in for WandbLogger that does nothing."""

    def __init__(self, *args, **kwargs):
        self.experiment = SimpleNamespace(unwatch=lambda model: None)

    def watch(self, model):
        pass


def _training_config(output_path: Path, **trainer_overrides) -> DictConfig:
    """
    Build the subset of the Hydra training config that the train functions read.

    Parameters
    ----------
    output_path : Path
        Directory for checkpoints and the saved config.
    **trainer_overrides
        Replacement entries for the trainer section.

    Returns
    -------
    DictConfig
        Training config with a tar dataloader.
    """
    trainer = dict(
        cuda=-1,
        num_nodes=1,
        num_epochs=1,
        gradient_clip_val=0.5,
        precision="32",
        output_path=str(output_path),
        project_name="test",
        wandb_id=None,
        checkpoint=None,
        fine_tune=False,
    )
    trainer.update(trainer_overrides)
    return OmegaConf.create(
        dict(trainer=trainer, dataloader=dict(type="tar", tar=dict(batch_size=3)))
    )


def _patch_training_module(module, monkeypatch) -> dict:
    """
    Replace the expensive or networked parts of a training module with fakes.

    Parameters
    ----------
    module : module
        starling.training.vae_train or starling.training.diffusion_train.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to undo the patches after the test.

    Returns
    -------
    dict
        Filled in while training runs: "trainer" (the _FakeTrainer) and
        "effective_batch_size" (what the data module was built with).
    """
    captured = {}

    def make_trainer(**kwargs):
        captured["trainer"] = _FakeTrainer(**kwargs)
        return captured["trainer"]

    def fake_data_module(cfg, effective_batch_size=None):
        captured["effective_batch_size"] = effective_batch_size
        return object()

    monkeypatch.setattr(module, "pl", SimpleNamespace(Trainer=make_trainer))
    monkeypatch.setattr(module, "wandb_init", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb", SimpleNamespace(finish=lambda: None))
    monkeypatch.setattr(module, "setup_data_module", fake_data_module)
    return captured


def test_vae_training_passes_gradient_clipping_and_world_size_batch(tmp_path, monkeypatch):
    """gradient_clip_val reaches the Trainer and the batch size uses the resolved world size."""
    captured = _patch_training_module(vae_train, monkeypatch)
    monkeypatch.setattr(vae_train, "WandbLogger", _FakeLogger)
    monkeypatch.setattr(vae_train, "setup_vae_model", lambda cfg: torch.nn.Linear(1, 1))
    cfg = _training_config(tmp_path / "vae_run")

    vae_train.train_vae.__wrapped__(cfg)

    assert captured["trainer"].kwargs["gradient_clip_val"] == 0.5
    assert captured["effective_batch_size"] == _FakeTrainer.world_size * 3


@pytest.mark.parametrize("has_last_checkpoint", [False, True])
def test_diffusion_training_resumes_from_last_checkpoint(tmp_path, monkeypatch, has_last_checkpoint):
    """Like vae_train, diffusion training resumes from output_path/last.ckpt when it exists."""
    captured = _patch_training_module(diffusion_train, monkeypatch)
    monkeypatch.setattr(diffusion_train, "setup_logger", lambda cfg, model: _FakeLogger())
    monkeypatch.setattr(diffusion_train, "setup_models", lambda cfg: torch.nn.Linear(1, 1))
    output_path = tmp_path / "ddpm_run"
    output_path.mkdir()
    if has_last_checkpoint:
        (output_path / "last.ckpt").write_bytes(b"")
    cfg = _training_config(output_path)

    diffusion_train.train_model.__wrapped__(cfg)

    assert captured["trainer"].fit_ckpt_path == ("last" if has_last_checkpoint else None)
    assert captured["effective_batch_size"] == _FakeTrainer.world_size * 3


def test_resume_checkpoint_respects_fine_tune_and_explicit_checkpoint(tmp_path):
    """Fine-tuning never resumes; an explicit checkpoint is resumed as given."""
    (tmp_path / "last.ckpt").write_bytes(b"")
    fine_tune = OmegaConf.create(dict(fine_tune=True, checkpoint="weights.ckpt"))
    explicit = OmegaConf.create(dict(fine_tune=False, checkpoint="weights.ckpt"))

    assert diffusion_train.get_resume_checkpoint(fine_tune, str(tmp_path)) is None
    assert diffusion_train.get_resume_checkpoint(explicit, str(tmp_path)) == "weights.ckpt"


@pytest.mark.parametrize("module", [vae_train, diffusion_train])
def test_effective_batch_size_uses_trainer_world_size(module):
    """Multi-device and multi-node CPU Trainers give world_size * batch_size."""
    quiet = dict(logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    three_devices = pl.Trainer(accelerator="cpu", devices=3, num_nodes=1, **quiet)
    two_nodes = pl.Trainer(accelerator="cpu", devices=2, num_nodes=2, **quiet)

    assert module.get_effective_batch_size(three_devices, 16) == 48
    assert module.get_effective_batch_size(two_nodes, 16) == 64
    with pytest.raises(ValueError, match="positive world size"):
        module.get_effective_batch_size(SimpleNamespace(world_size=0), 16)


@pytest.mark.parametrize("module", [vae_train, diffusion_train])
def test_h5_dataloader_type_is_rejected_clearly(module):
    """The h5 branch cannot work with the shipped configs, so it must say so."""
    cfg = OmegaConf.create(dict(dataloader=dict(type="h5", h5=dict(train="train.csv"))))

    with pytest.raises(NotImplementedError, match="'h5' is not supported"):
        module.setup_data_module(cfg)


def test_diffusion_vit_context_dim_follows_sequence_encoder(monkeypatch):
    """The ViT cross-attention must take the sequence encoder's embed_dim as context."""
    # One ViT layer keeps the model small; the other released-model values are kept
    monkeypatch.setattr(diffusion_train, "VIT_NUM_LAYERS", 1)
    sequence_encoder_dim = 32
    cfg = OmegaConf.create(
        dict(
            trainer=dict(checkpoint=None, fine_tune=False),
            sequence_encoder=dict(num_layers=1, embed_dim=sequence_encoder_dim, num_heads=2),
            diffusion=dict(type="discrete", discrete=dict(distance_map_encoder=None, timesteps=10)),
        )
    )

    diffusion_model = diffusion_train.setup_models(cfg)

    cross_attention = diffusion_model.model.transformer_layers[0].cross_attention
    assert cross_attention.key_proj.in_features == sequence_encoder_dim
    assert cross_attention.key_proj.out_features == diffusion_train.VIT_EMBED_DIM


# ---------------------------------------------------------------------------
# VAE evaluation
# ---------------------------------------------------------------------------


class _StubVAE(torch.nn.Module):
    """
    Tiny VAE stand-in whose posterior mean is the input and whose decoder doubles it.

    The posterior has unit variance, so decoding a random sample would give a
    different answer every call; forward() fails so any use of it is caught.
    """

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.grad_enabled_in_decode = None

    def encode(self, data):
        return DiagonalGaussianDistribution(torch.cat([data, torch.zeros_like(data)], dim=1))

    def decode(self, latents):
        self.grad_enabled_in_decode = torch.is_grad_enabled()
        return latents * self.scale

    def forward(self, data):
        raise AssertionError("reconstruction must decode the posterior mean, not model(x)")


def test_reconstruct_decodes_posterior_mean_without_autograd():
    """evaluate_vae.reconstruct is deterministic and builds no autograd graph."""
    model = _StubVAE()
    distance_maps = torch.rand(3, 1, MAP_SIZE, MAP_SIZE)

    first = evaluate_vae.reconstruct(model, distance_maps)
    second = evaluate_vae.reconstruct(model, distance_maps)

    np.testing.assert_allclose(first, 2 * distance_maps.numpy().squeeze())
    np.testing.assert_array_equal(first, second)
    assert model.grad_enabled_in_decode is False


def test_evaluate_vae_imports_without_tabulate(monkeypatch):
    """tabulate is not a declared dependency, so importing the module must not need it."""
    monkeypatch.setitem(sys.modules, "tabulate", None)
    monkeypatch.delitem(sys.modules, "starling.inference.evaluate_vae")
    # Re-importing rebinds the package attribute, so restore it after the test too
    monkeypatch.setattr(starling.inference, "evaluate_vae", evaluate_vae)

    module = importlib.import_module("starling.inference.evaluate_vae")

    assert callable(module.reconstruct)


# ---------------------------------------------------------------------------
# data_wrangler helpers
# ---------------------------------------------------------------------------


def test_one_hot_encode_pads_different_length_sequences():
    """Shorter sequences are right-padded with the "0" padding token."""
    encoded = data_wrangler.one_hot_encode(["AC", "ACD"])
    padded_by_hand = data_wrangler.one_hot_encode(["AC0", "ACD"])

    assert encoded.shape == (2, 3, 21)
    np.testing.assert_array_equal(encoded, padded_by_hand)
    np.testing.assert_array_equal(encoded.sum(axis=-1), np.ones((2, 3)))


def test_maxpad_pads_and_refuses_to_shrink():
    """MaxPad pads to the requested shape and raises rather than returning a larger map."""
    target_shape = (384, 384)

    assert data_wrangler.MaxPad(np.ones((10, 10)), target_shape).shape == target_shape
    with pytest.raises(ValueError, match="smaller shape"):
        data_wrangler.MaxPad(np.ones((400, 400)), target_shape)
    assert "shape : tuple[int, int]" in data_wrangler.MaxPad.__doc__


def test_load_hdf5_compressed_rejects_out_of_range_frame(tmp_path):
    """Asking for a frame that does not exist raises instead of returning every frame."""
    path = tmp_path / "frames.h5"
    distance_maps = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
    with h5py.File(path, "w") as handle:
        handle["dm"] = distance_maps
        handle["seq"] = np.arange(4)

    data = data_wrangler.load_hdf5_compressed(path, frame=2)
    np.testing.assert_array_equal(data["dm"], distance_maps[2])
    np.testing.assert_array_equal(data["seq"], np.arange(4))

    with pytest.raises(IndexError, match="frame 10 is out of range"):
        data_wrangler.load_hdf5_compressed(path, frame=10)
