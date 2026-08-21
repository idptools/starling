"""
Tests for offline/local resolution of model weights and search artifacts.

These exercise the resolution logic in ``starling.configs`` without touching
the network: every test that could otherwise download sets
``STARLING_OFFLINE`` so a failure surfaces as ``FileNotFoundError`` rather
than a (slow, flaky) network call.
"""

import os

import pytest

from starling import configs

URL = "https://example.invalid/releases/download/v0/fake_weights.ckpt"
NAME = "fake_weights.ckpt"


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """Point DEFAULT_MODEL_DIR and TORCH_HOME at empty temp directories."""
    model_dir = tmp_path / "starling_weights"
    torch_home = tmp_path / "torch_home"
    model_dir.mkdir()
    monkeypatch.setattr(configs, "DEFAULT_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("TORCH_HOME", str(torch_home))
    monkeypatch.setenv("STARLING_OFFLINE", "1")
    return model_dir, torch_home


def test_local_path_is_returned_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("STARLING_OFFLINE", "1")
    local = tmp_path / "weights.ckpt"
    local.write_bytes(b"x")
    assert configs.resolve_weights_path(str(local), URL) == str(local)


def test_tilde_in_local_path_is_expanded(monkeypatch):
    monkeypatch.setenv("STARLING_OFFLINE", "1")
    resolved = configs.resolve_weights_path("~/some/weights.ckpt", URL)
    assert resolved == os.path.expanduser("~/some/weights.ckpt")


def test_none_falls_back_to_default_url_and_local_dir(isolated_dirs):
    model_dir, _ = isolated_dirs
    target = model_dir / NAME
    target.write_bytes(b"x")
    assert configs.resolve_weights_path(None, URL) == str(target)


def test_starling_weights_dir_is_honoured(isolated_dirs):
    model_dir, _ = isolated_dirs
    target = model_dir / NAME
    target.write_bytes(b"x")
    assert configs.resolve_weights_path(URL, URL) == str(target)


def test_torch_hub_cache_is_honoured(isolated_dirs):
    _, torch_home = isolated_dirs
    ckpt_dir = torch_home / "hub" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    target = ckpt_dir / NAME
    target.write_bytes(b"x")
    assert configs.resolve_weights_path(URL, URL) == str(target)


def test_starling_weights_dir_wins_over_hub_cache(isolated_dirs):
    model_dir, torch_home = isolated_dirs
    ckpt_dir = torch_home / "hub" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / NAME).write_bytes(b"hub")
    (model_dir / NAME).write_bytes(b"local")
    assert configs.resolve_weights_path(URL, URL) == str(model_dir / NAME)


def test_offline_missing_weights_raises_with_search_locations(isolated_dirs):
    model_dir, torch_home = isolated_dirs
    with pytest.raises(FileNotFoundError) as excinfo:
        configs.resolve_weights_path(URL, URL)
    msg = str(excinfo.value)
    assert "STARLING_OFFLINE" in msg
    assert str(model_dir / NAME) in msg
    assert str(torch_home / "hub" / "checkpoints" / NAME) in msg
    assert "STARLING_ENCODER_PATH" in msg


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("off", False),
    ],
)
def test_is_offline_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("STARLING_OFFLINE", value)
    assert configs.is_offline() is expected


def test_is_offline_unset(monkeypatch):
    monkeypatch.delenv("STARLING_OFFLINE", raising=False)
    assert configs.is_offline() is False


def test_torch_hub_checkpoint_dir_respects_torch_home(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    assert configs.torch_hub_checkpoint_dir() == str(tmp_path / "hub" / "checkpoints")


def test_search_artifact_offline_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("STARLING_OFFLINE", "1")
    dest = tmp_path / "index.faiss"
    with pytest.raises(FileNotFoundError) as excinfo:
        configs._download_if_missing(
            "https://example.invalid/index.faiss", str(dest), ""
        )
    assert "STARLING_FAISS_INDEX_PATH" in str(excinfo.value)


def test_search_artifact_offline_present_bad_md5_is_used(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STARLING_OFFLINE", "1")
    dest = tmp_path / "index.faiss"
    dest.write_bytes(b"not the real index")
    # must not raise and must not attempt a download
    configs._download_if_missing(
        "https://example.invalid/index.faiss", str(dest), "0" * 32
    )
    assert dest.read_bytes() == b"not the real index"
    assert "STARLING_OFFLINE" in capsys.readouterr().out


def test_search_artifact_present_good_md5_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("STARLING_OFFLINE", raising=False)
    dest = tmp_path / "index.faiss"
    dest.write_bytes(b"abc")
    configs._download_if_missing(
        "https://example.invalid/index.faiss", str(dest), configs._md5_file(str(dest))
    )
    assert dest.read_bytes() == b"abc"
