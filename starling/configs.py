import importlib.util
import os

from starling.utilities import fix_ref_to_home

# stand-alone default parameters
# NB: you can overwrite these by adding a configs.py file to ~/.starling_weights/
DEFAULT_MODEL_DIR = os.path.join(
    os.path.expanduser(os.path.join("~/", ".starling_weights"))
)
DEFAULT_ENCODE_WEIGHTS = "model-kernel-epoch=99-epoch_val_loss=1.72.ckpt"
DEFAULT_DDPM_WEIGHTS = "model-kernel-epoch=47-epoch_val_loss=0.03.ckpt"
DEFAULT_NUMBER_CONFS = 200
DEFAULT_BATCH_SIZE = 100
DEFAULT_STEPS = 25
DEFAULT_MDS_NUM_INIT = 4
DEFAULT_STRUCTURE_GEN = "mds"
CONVERT_ANGSTROM_TO_NM = 10
MAX_SEQUENCE_LENGTH = 384  # set longest sequence the model can work on


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
GITHUB_ENCODER_URL = "https://github.com/idptools/starling/releases/download/v1.0.0/model-kernel-epoch.99-epoch_val_loss.1.72.ckpt"
GITHUB_DDPM_URL = "https://github.com/idptools/starling/releases/download/v1.0.0/model-kernel-epoch.47-epoch_val_loss.0.03.ckpt"

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
