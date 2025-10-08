import importlib.util
import os

from starling.utilities import fix_ref_to_home

# stand-alone default parameters
# NB: you can overwrite these by adding a configs.py file to ~/.starling_weights/
DEFAULT_MODEL_DIR = os.path.join(
    os.path.expanduser(os.path.join("~/", ".starling_weights"))
)
# DEFAULT_ENCODE_WEIGHTS = "model-kernel-epoch=99-epoch_val_loss=1.72.ckpt"
# DEFAULT_DDPM_WEIGHTS = "model-kernel-epoch=47-epoch_val_loss=0.03.ckpt"

DEFAULT_ENCODE_WEIGHTS = "vae.ckpt"
DEFAULT_DDPM_WEIGHTS = "diffusion_model.ckpt"
DEFAULT_NUMBER_CONFS = 400
DEFAULT_BATCH_SIZE = 100
DEFAULT_STEPS = 30
DEFAULT_MDS_NUM_INIT = 4
DEFAULT_STRUCTURE_GEN = "mds"
CONVERT_ANGSTROM_TO_NM = 10
MAX_SEQUENCE_LENGTH = 380  # set longest sequence the model can work on
DEFAULT_IONIC_STRENGTH = 150  # default ionic strength in mM
DEFAULT_SAMPLER = "ddim"  # default sampler for diffusion model

# Model compilation settings
TORCH_COMPILATION = {
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

# default paths to the model weights
DEFAULT_ENCODER_WEIGHTS_PATH = fix_ref_to_home(
    os.path.join(DEFAULT_MODEL_DIR, DEFAULT_ENCODE_WEIGHTS)
)
DEFAULT_DDPM_WEIGHTS_PATH = fix_ref_to_home(
    os.path.join(DEFAULT_MODEL_DIR, DEFAULT_DDPM_WEIGHTS)
)

# Github Releases URLs for model weights
GITHUB_ENCODER_URL = (
    "https://github.com/idptools/starling/releases/download/v2.0.0-alpha/vae.ckpt"
)
GITHUB_DDPM_URL = "https://github.com/idptools/starling/releases/download/v2.0.0-alpha/diffusion_model.ckpt"

# Update default paths to check Hub first
DEFAULT_ENCODER_WEIGHTS_PATH = os.environ.get(
    "STARLING_ENCODER_PATH", GITHUB_ENCODER_URL
)
DEFAULT_DDPM_WEIGHTS_PATH = os.environ.get("STARLING_DDPM_PATH", GITHUB_DDPM_URL)

# Set the default number of CPUs to use
DEFAULT_CPU_COUNT_MDS = min(DEFAULT_MDS_NUM_INIT, os.cpu_count())

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

# TODO: update these to work once we have real Zenodo DOIs / file URLs
# may have to migrate to another hosting solution if Zenodo doesn't support
# direct file linking without going through their UI? or maybe theres a rest api?
ZENODO_FAISS_INDEX_URL = os.environ.get(
    "STARLING_ZENODO_FAISS_URL",
    "PLACEHOLDER",
)
ZENODO_SEQSTORE_URL = os.environ.get(
    "STARLING_ZENODO_SEQSTORE_URL",
    "PLACEHOLDER",
)
ZENODO_MANIFEST_URL = os.environ.get(
    "STARLING_ZENODO_MANIFEST_URL",
    "PLACEHOLDER",
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

# Optional SHA256 hashes (empty by default). Set via env for integrity checking.
FAISS_INDEX_SHA256 = os.environ.get("STARLING_FAISS_INDEX_SHA256", "")
SEQSTORE_SHA256 = os.environ.get("STARLING_SEQSTORE_SHA256", "")
MANIFEST_SHA256 = os.environ.get("STARLING_FAISS_MANIFEST_SHA256", "")


def _sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_if_missing(url: str, dest: str, expected_sha256: str = "") -> None:
    """Download a file to a temporary path then atomically publish.
    Writes to dest+'.part' first; on success (and optional hash verify) renames to dest.
    Cleans up partial file on failure or hash mismatch.
    """
    if not url or "PLACEHOLDER" in url:
        return
    need = True
    if os.path.exists(dest):
        if expected_sha256:
            try:
                if _sha256_file(dest) == expected_sha256.lower():
                    need = False
            except Exception:
                pass
        else:
            need = False
    if not need:
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    try:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
        import urllib.request

        print(f"[Starling Search] Downloading {url} -> {dest}")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                if not chunk:
                    break
                f.write(chunk)
        # Hash verify before publish
        if expected_sha256:
            try:
                got = _sha256_file(tmp)
                if got.lower() != expected_sha256.lower():
                    print(
                        f"[Starling Search] SHA256 mismatch (expected {expected_sha256} got {got}); discarding"
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
    except Exception as e:
        print(f"[Starling Search] Download failed ({url}): {e}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return


def ensure_search_artifacts(download: bool = True) -> tuple[str, str, str]:
    """Ensure FAISS index + sequence store + manifest exist locally.

    Attempts download from Zenodo-like URLs if missing and download=True.
    Returns (index_path, seqstore_path, manifest_path) regardless of existence.
    The caller should still validate accessibility (e.g. faiss.read_index).
    """
    if download:
        _download_if_missing(
            ZENODO_FAISS_INDEX_URL, DEFAULT_FAISS_INDEX_PATH, FAISS_INDEX_SHA256
        )
        _download_if_missing(
            ZENODO_SEQSTORE_URL, DEFAULT_SEQSTORE_DB_PATH, SEQSTORE_SHA256
        )
        _download_if_missing(
            ZENODO_MANIFEST_URL, DEFAULT_FAISS_MANIFEST_PATH, MANIFEST_SHA256
        )
    return (
        DEFAULT_FAISS_INDEX_PATH,
        DEFAULT_SEQSTORE_DB_PATH,
        DEFAULT_FAISS_MANIFEST_PATH,
    )
