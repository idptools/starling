import importlib.util
import os
from typing import TypedDict

from starling.utilities import fix_ref_to_home

# stand-alone default parameters
# NB: you can overwrite these by adding a configs.py file to ~/.starling_weights/
DEFAULT_MODEL_DIR = os.path.join(
    os.path.expanduser(os.path.join("~/", ".starling_weights"))
)
# DEFAULT_ENCODE_WEIGHTS = "model-kernel-epoch=99-epoch_val_loss=1.72.ckpt"
# DEFAULT_DDPM_WEIGHTS = "model-kernel-epoch=47-epoch_val_loss=0.03.ckpt"

DEFAULT_ENCODE_WEIGHTS = "STARLING_v2.0.0_ViT_VAE_2025_10_14.ckpt"
DEFAULT_DDPM_WEIGHTS = "STARLING_v2.0.0_ViT_DDPM_2025_10_14.ckpt"
DEFAULT_NUMBER_CONFS = 400
DEFAULT_BATCH_SIZE = 100
DEFAULT_STEPS = 30
DEFAULT_STRUCTURE_GEN = "mds"
CONVERT_ANGSTROM_TO_NM = 10
MAX_SEQUENCE_LENGTH = 380  # set longest sequence the model can work on
DEFAULT_IONIC_STRENGTH = 150  # default ionic strength in mM
DEFAULT_SAMPLER = "ddim"  # default sampler for diffusion model

# Model compilation settings
class TorchCompilationOptions(TypedDict):
    mode: str
    fullgraph: bool
    backend: str
    dynamic: bool | None


class TorchCompilationConfig(TypedDict):
    enabled: bool
    options: TorchCompilationOptions


TORCH_COMPILATION: TorchCompilationConfig = {
    "enabled": False,
    "options": {
        "mode": "default",  # Options: "default", "reduce-overhead", "max-autotune"
        "fullgraph": True,  # Whether to use the full graph for compilation
        "backend": "inductor",  # Default PyTorch backend
        "dynamic": None,  # Whether to handle dynamic shapes
    },
}


# model model-kernel-epoch=47-epoch_val_loss=0.03.ckpt has  a UNET_LABELS_DIM of 512
# model model-kernel-epoch=47-epoch_val_loss=0.03.ckpt has a UNET_LABELS_DIM of 384
UNET_LABELS_DIM = 512

# Path to user config file
USER_CONFIG_PATH = os.path.expanduser(
    os.path.join("~/", ".starling_weights", "configs.py")
)


##
## The code block below lets us over-ride default values based on the configs.py file in the
## ~/.starling_weights directory
##


def load_user_config():
    """Load user configuration if the file exists and override default values."""
    if os.path.exists(USER_CONFIG_PATH):
        spec = importlib.util.spec_from_file_location("user_config", USER_CONFIG_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load STARLING configuration from {USER_CONFIG_PATH}")
        user_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_config)

        for key, value in vars(user_config).items():
            if not key.startswith("__") and key in globals():
                old_value = globals()[key]
                globals()[key] = value
                print(f"[Starling Config] Overriding {key}: {old_value} → {value}")


# Load user-defined config if available
load_user_config()

### Derived default values

# Github Releases URLs for model weights
GITHUB_ENCODER_URL = f"https://github.com/idptools/starling/releases/download/v2.0.0/{DEFAULT_ENCODE_WEIGHTS}"
GITHUB_DDPM_URL = f"https://github.com/idptools/starling/releases/download/v2.0.0/{DEFAULT_DDPM_WEIGHTS}"


def _env_flag(name: str) -> bool:
    """
    Interpret an environment variable as a boolean switch.

    Parameters
    ----------
    name : str
        Name of the environment variable.

    Returns
    -------
    bool
        True if the variable is set to something other than an empty string,
        ``0``, ``false``, ``no`` or ``off`` (case-insensitive).
    """
    value = os.environ.get(name, "")
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def is_offline() -> bool:
    """
    Return True if STARLING has been told never to touch the network.

    Offline mode is enabled by setting the ``STARLING_OFFLINE`` environment
    variable (e.g. ``export STARLING_OFFLINE=1``). In offline mode STARLING
    will only ever use locally available model weights and search artifacts,
    and raises a ``FileNotFoundError`` (listing every location it looked in)
    rather than attempting a download if something is missing.

    Returns
    -------
    bool
        True if ``STARLING_OFFLINE`` is set to a truthy value.
    """
    return _env_flag("STARLING_OFFLINE")


def torch_hub_checkpoint_dir() -> str:
    """
    Return the directory torch.hub uses to cache downloaded checkpoints.

    This is ``$TORCH_HOME/hub/checkpoints`` if ``TORCH_HOME`` is set, and
    ``~/.cache/torch/hub/checkpoints`` otherwise. We resolve this lazily
    (rather than at import) so that a ``TORCH_HOME`` set after import is
    still honoured, and so importing ``starling.configs`` does not require
    torch.

    Returns
    -------
    str
        Absolute path to the checkpoint cache directory.
    """
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home is None:
        xdg = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        torch_home = os.path.join(xdg, "torch")
    return os.path.join(os.path.expanduser(torch_home), "hub", "checkpoints")


def candidate_weights_paths(filename: str) -> list[str]:
    """
    Return the local locations searched (in order) for a weights file.

    Parameters
    ----------
    filename : str
        Basename of the checkpoint file, e.g. ``DEFAULT_ENCODE_WEIGHTS``.

    Returns
    -------
    list of str
        Candidate absolute paths, highest priority first:

        1. ``DEFAULT_MODEL_DIR`` (``~/.starling_weights`` by default, which can
           be changed via ``~/.starling_weights/configs.py``).
        2. The torch.hub checkpoint cache (where STARLING saves weights it has
           downloaded itself).
    """
    return [
        fix_ref_to_home(os.path.join(DEFAULT_MODEL_DIR, filename)),
        os.path.join(torch_hub_checkpoint_dir(), filename),
    ]


def resolve_weights_path(path_or_url: str | None, default_url: str) -> str:
    """
    Resolve a model weights reference to a local file, downloading if allowed.

    The resolution order is:

    1. If ``path_or_url`` is a local path (anything that does not start with
       ``http``), it is returned as-is.
    2. Otherwise the basename of the URL is looked for in each of the local
       directories returned by :func:`candidate_weights_paths`; the first
       file that exists wins.
    3. If nothing is found locally and ``STARLING_OFFLINE`` is not set, the
       file is downloaded from the URL into the torch.hub checkpoint cache.
    4. If nothing is found locally and ``STARLING_OFFLINE`` *is* set, a
       ``FileNotFoundError`` is raised listing every location that was
       searched.

    Parameters
    ----------
    path_or_url : str or None
        A local filesystem path, an ``http(s)`` URL, or None (in which case
        ``default_url`` is used).
    default_url : str
        URL to fall back to when ``path_or_url`` is None.

    Returns
    -------
    str
        Path to a local weights file.

    Raises
    ------
    FileNotFoundError
        If the weights cannot be located locally and downloading is not
        permitted (``STARLING_OFFLINE`` set) or fails.
    """
    ref = path_or_url or default_url

    if not ref.startswith("http"):
        return fix_ref_to_home(ref)

    filename = os.path.basename(ref)
    candidates = candidate_weights_paths(filename)
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    searched = "\n".join(f"  - {c}" for c in candidates)
    if is_offline():
        raise FileNotFoundError(
            f"STARLING_OFFLINE is set and the weights file '{filename}' was not "
            f"found in any of the following locations:\n{searched}\n"
            f"Download it from {ref} on a machine with internet access and copy "
            f"it to one of the directories above (make sure it is readable by the "
            f"user running STARLING), or point the STARLING_ENCODER_PATH / "
            f"STARLING_DDPM_PATH environment variables at the file directly."
        )

    import torch

    cache_dir = torch_hub_checkpoint_dir()
    os.makedirs(cache_dir, exist_ok=True)
    cached_path = os.path.join(cache_dir, filename)
    try:
        torch.hub.download_url_to_file(ref, cached_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not download '{filename}' from {ref} ({e.__class__.__name__}: {e}). "
            f"If this machine has no internet access, download the file elsewhere "
            f"and copy it to one of:\n{searched}\n"
            f"then set STARLING_OFFLINE=1 to suppress download attempts."
        ) from e
    return cached_path


# Default weight references. These may be a local path or a URL; either way
# they are resolved (locally first, then by download) via resolve_weights_path()
# at model-load time. The STARLING_ENCODER_PATH / STARLING_DDPM_PATH
# environment variables take precedence over everything else.
DEFAULT_ENCODER_WEIGHTS_PATH = os.environ.get(
    "STARLING_ENCODER_PATH", GITHUB_ENCODER_URL
)
DEFAULT_DDPM_WEIGHTS_PATH = os.environ.get("STARLING_DDPM_PATH", GITHUB_DDPM_URL)

# define valid amino acids
VALID_AA = "ACDEFGHIKLMNPQRSTVWY"

# define conversion dictionaries for AAs
AA_THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

AA_ONE_TO_THREE = {}
for x in AA_THREE_TO_ONE:
    AA_ONE_TO_THREE[AA_THREE_TO_ONE[x]] = x

# ---------------------------------------------------------------------------
# Search (FAISS + SQLite) default configuration & lazy fetch
# ---------------------------------------------------------------------------
# Directory for cached search artifacts (separate from model weights to allow lighter syncs)
DEFAULT_SEARCH_DIR = os.path.expanduser(os.path.join("~", ".starling_search"))

# Default artifact filenames (can be overridden via user config or env)
DEFAULT_FAISS_INDEX_NAME = (
    "ensemble_search_gpu_nlist_32768_m_64_nbits_8_use_opq_True_compressed_False.faiss"
)
DEFAULT_SEQSTORE_NAME = DEFAULT_FAISS_INDEX_NAME + ".seqs.sqlite"
DEFAULT_MANIFEST_NAME = DEFAULT_FAISS_INDEX_NAME + ".manifest.json"

# Environment variable overrides (paths OR HTTP(S) URLs)
ENV_FAISS_INDEX_PATH = os.environ.get("STARLING_FAISS_INDEX_PATH")
ENV_SEQSTORE_PATH = os.environ.get("STARLING_SEQSTORE_PATH")
ENV_MANIFEST_PATH = os.environ.get("STARLING_FAISS_MANIFEST_PATH")

ZENODO_FAISS_INDEX_URL = os.environ.get(
    "STARLING_ZENODO_FAISS_URL",
    "https://zenodo.org/records/17342150/files/ensemble_search_gpu_nlist_32768_m_64_nbits_8_use_opq_True_compressed_False.faiss?download=1",
)
ZENODO_SEQSTORE_URL = os.environ.get(
    "STARLING_ZENODO_SEQSTORE_URL",
    "https://zenodo.org/records/17342150/files/ensemble_search_gpu_nlist_32768_m_64_nbits_8_use_opq_True_compressed_False.faiss.seqs.sqlite?download=1",
)
ZENODO_MANIFEST_URL = os.environ.get(
    "STARLING_ZENODO_MANIFEST_URL",
    "https://zenodo.org/records/17342150/files/ensemble_search_gpu_nlist_32768_m_64_nbits_8_use_opq_True_compressed_False.faiss.manifest.json?download=1",
)

# Resolved local cache paths (before existence check)
DEFAULT_FAISS_INDEX_PATH = ENV_FAISS_INDEX_PATH or os.path.join(
    DEFAULT_SEARCH_DIR, DEFAULT_FAISS_INDEX_NAME
)
DEFAULT_SEQSTORE_DB_PATH = ENV_SEQSTORE_PATH or os.path.join(
    DEFAULT_SEARCH_DIR, DEFAULT_SEQSTORE_NAME
)
DEFAULT_FAISS_MANIFEST_PATH = ENV_MANIFEST_PATH or os.path.join(
    DEFAULT_SEARCH_DIR, DEFAULT_MANIFEST_NAME
)


FAISS_INDEX_MD5 = (
    os.environ.get("STARLING_FAISS_INDEX_MD5") or "e4a72e12b2f9cdabd8ec4f8207f3d28d"
)
SEQSTORE_MD5 = (
    os.environ.get("STARLING_SEQSTORE_MD5") or "ade24690e7962768eee1acbb4f95904c"
)
MANIFEST_MD5 = (
    os.environ.get("STARLING_FAISS_MANIFEST_MD5") or "f0057554e3303b3f2e7b4e2fd3aad70a"
)


def _md5_file(path: str) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_expected_md5(expected: str) -> str:
    digest = expected.strip().lower()
    if not digest:
        return ""
    if digest.startswith("md5:"):
        digest = digest.split(":", 1)[1]
    return digest


def _download_if_missing(url: str, dest: str, expected_checksum: str = "") -> None:
    """Download a file to a temporary path then atomically publish.
    Writes to dest+'.part' first; on success (and optional hash verify) renames to dest.
    Cleans up partial file on failure or hash mismatch.
    """
    if not url or "PLACEHOLDER" in url:
        return
    need = True
    expected_md5 = _normalize_expected_md5(expected_checksum)
    if os.path.exists(dest):
        if expected_md5:
            try:
                if _md5_file(dest) == expected_md5:
                    need = False
            except Exception:
                pass
        else:
            need = False
    if not need:
        return
    if is_offline():
        if os.path.exists(dest):
            # present but failed the checksum; trust the user's file rather
            # than attempt a download we have been told not to make
            print(
                f"[Starling Search] STARLING_OFFLINE is set; using {dest} even "
                f"though its MD5 does not match the expected value."
            )
            return
        raise FileNotFoundError(
            f"STARLING_OFFLINE is set and the search artifact '{os.path.basename(dest)}' "
            f"was not found at {dest}. Download it from {url} on a machine with "
            f"internet access and copy it to that location, or point the "
            f"STARLING_FAISS_INDEX_PATH / STARLING_SEQSTORE_PATH / "
            f"STARLING_FAISS_MANIFEST_PATH environment variables at the files."
        )
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    resume_bytes = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    from urllib import error, request

    while True:
        headers = {}
        if resume_bytes:
            headers["Range"] = f"bytes={resume_bytes}-"
        req = request.Request(url, headers=headers)
        try:
            resp = request.urlopen(req)
            break
        except error.HTTPError as e:
            if resume_bytes and e.code == 416:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                resume_bytes = 0
                continue
            raise

    total_size = resp.getheader("Content-Length")
    if total_size is not None:
        total_size = int(total_size)
        if getattr(resp, "status", None) == 206:
            total_size += resume_bytes

    mode = "ab" if resume_bytes and getattr(resp, "status", None) == 206 else "wb"
    if mode == "wb" and resume_bytes:
        resume_bytes = 0

    print(
        f"[Starling Search] Downloading {url} -> {dest}"
        + (" (resuming)" if resume_bytes else "")
    )

    chunk_size = 4 << 20
    from tqdm import tqdm

    progress = tqdm(
        total=total_size,
        initial=resume_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=os.path.basename(dest),
    )

    try:
        with open(tmp, mode) as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                progress.update(len(chunk))
    finally:
        progress.close()
        resp.close()
    # Hash verify before publish
    if expected_md5:
        try:
            got = _md5_file(tmp)
            if got.lower() != expected_md5.lower():
                print(
                    f"[Starling Search] MD5 mismatch (expected {expected_md5} got {got}); discarding"
                )
                try:
                    os.remove(tmp)
                except Exception:
                    pass
                return
        except Exception as e:
            print(f"[Starling Search] Hash check failed: {e}")
            # proceed without deleting; still publish
    os.replace(tmp, dest)


def ensure_search_artifacts(download: bool = True) -> tuple[str, str, str]:
    """Ensure FAISS index, sequence store, and manifest are present locally.

    Attempts to download from the configured URLs when files are missing and
    ``download`` is True. Returns the resolved paths regardless of existence.
    """
    if download:
        _download_if_missing(
            ZENODO_FAISS_INDEX_URL,
            DEFAULT_FAISS_INDEX_PATH,
            FAISS_INDEX_MD5,
        )
        _download_if_missing(
            ZENODO_SEQSTORE_URL,
            DEFAULT_SEQSTORE_DB_PATH,
            SEQSTORE_MD5,
        )
        _download_if_missing(
            ZENODO_MANIFEST_URL,
            DEFAULT_FAISS_MANIFEST_PATH,
            MANIFEST_MD5,
        )
    return (
        DEFAULT_FAISS_INDEX_PATH,
        DEFAULT_SEQSTORE_DB_PATH,
        DEFAULT_FAISS_MANIFEST_PATH,
    )
