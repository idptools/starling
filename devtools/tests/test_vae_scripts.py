"""Regression tests for the VAE tar-shard scripts, run on tiny shards with a stub VAE."""

import importlib.util
import io
import re
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import webdataset as wds

from starling.data.distributions import DiagonalGaussianDistribution

SCRIPTS = Path(__file__).parents[1] / "scripts"

# Two shards of five samples with batches of three, so every pass ends on a
# partial batch that partial=False used to drop
N_SHARDS = 2
SAMPLES_PER_SHARD = 5
N_SAMPLES = N_SHARDS * SAMPLES_PER_SHARD
BATCH_SIZE = 3

# Toy distance-map size and padded sequence length
MAP_SIZE = 6
SEQUENCE_LENGTH = 8


def _load_script(name: str):
    """
    Import a devtools script as a module.

    Parameters
    ----------
    name : str
        Script file name without the .py suffix.

    Returns
    -------
    module
        The imported script.
    """
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _write_shards(shard_dir: Path) -> None:
    """
    Write distance-map shards whose sample keys end in an ionic strength.

    Parameters
    ----------
    shard_dir : Path
        Directory to write the .tar shards into.
    """
    shard_dir.mkdir(parents=True)
    for shard in range(N_SHARDS):
        with wds.TarWriter(str(shard_dir / f"shard_{shard:04d}.tar")) as sink:
            for index in range(SAMPLES_PER_SHARD):
                sequence = np.zeros(SEQUENCE_LENGTH, dtype=np.int64)
                sequence[: 3 + index] = 1
                distance_map = np.full((MAP_SIZE, MAP_SIZE), 3.8, dtype=np.float32)
                np.fill_diagonal(distance_map, 0)
                sink.write(
                    {
                        "__key__": f"s{shard}_i{index}_150mM",
                        "distance_map.npz": _npz_bytes(distance_map),
                        "sequence.npz": _npz_bytes(sequence),
                    }
                )


def _single_process_webdataset() -> SimpleNamespace:
    """
    Build a stand-in for the webdataset module whose WebLoader uses no worker processes.

    The scripts hard-code worker counts; running in-process keeps the test fast and
    still exercises the per-pipeline partial-batch behaviour.

    Returns
    -------
    SimpleNamespace
        Exposes WebDataset, TarWriter and an in-process WebLoader.
    """

    def web_loader(dataset, **kwargs):
        kwargs["num_workers"] = 0
        kwargs.pop("prefetch_factor", None)
        return wds.WebLoader(dataset, **kwargs)

    return SimpleNamespace(WebDataset=wds.WebDataset, WebLoader=web_loader, TarWriter=wds.TarWriter)


class _StubVAE(torch.nn.Module):
    """
    Tiny VAE stand-in whose posterior mean is the input and whose decoder is the identity.

    forward() fails, so any reconstruction from a random posterior sample is caught.
    """

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def encode(self, data):
        return DiagonalGaussianDistribution(torch.cat([data, torch.zeros_like(data)], dim=1))

    def decode(self, latents):
        return latents * self.scale

    def forward(self, data):
        raise AssertionError("reconstruction must decode the posterior mean, not model(x)")


def _patch_script(script, monkeypatch, shard_dir: Path) -> None:
    """
    Point a script at the synthetic shards, the stub VAE and an in-process loader.

    Parameters
    ----------
    script : module
        The imported script.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to undo the patches after the test.
    shard_dir : Path
        Directory holding the synthetic shards.
    """
    monkeypatch.setattr(script, "TARDIR", str(shard_dir))
    monkeypatch.setattr(script, "BATCH_SIZE", BATCH_SIZE)
    monkeypatch.setattr(script, "load_model", lambda model_path: _StubVAE())
    monkeypatch.setattr(script, "wds", _single_process_webdataset())


def test_extract_latents_tar_writes_every_sample(tmp_path, monkeypatch):
    """The final partial batch must be kept so every input sample gets a latent."""
    script = _load_script("extract_latents_tar")
    _write_shards(tmp_path / "shards")
    output_dir = tmp_path / "latents"
    _patch_script(script, monkeypatch, tmp_path / "shards")
    monkeypatch.setattr(script, "OUTDIR", str(output_dir))

    script.main()

    written_keys = set()
    for shard in output_dir.glob("*.tar"):
        with tarfile.open(shard) as archive:
            written_keys.update(name.split(".")[0] for name in archive.getnames())
    assert len(written_keys) == N_SAMPLES


def test_vae_test_saves_statistics_for_every_sample(tmp_path, monkeypatch):
    """Statistics are saved after the loop and cover every sample, including the partial batch."""
    script = _load_script("vae_test")
    _write_shards(tmp_path / "shards")
    _patch_script(script, monkeypatch, tmp_path / "shards")
    monkeypatch.chdir(tmp_path)

    script.main()

    statistics = (tmp_path / script.STATS_FILE).read_text()
    sample_counts = [int(count) for count in re.findall(r"Samples: (\d+)", statistics)]
    average_losses = [float(loss) for loss in re.findall(r"Average Loss: ([0-9.]+)", statistics)]
    assert sum(sample_counts) == N_SAMPLES
    # The stub decoder returns the posterior mean unchanged, so the loss is exactly zero
    assert average_losses == [0.0] * len(average_losses)
