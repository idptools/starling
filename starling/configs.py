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

DEFAULT_ENCODE_WEIGHTS = "STARLING_v2.0.0_ViT_VAE_2025_10_14.ckpt"
DEFAULT_DDPM_WEIGHTS = "STARLING_v2.0.0_ViT_DDPM_2025_10_14.ckpt"
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
