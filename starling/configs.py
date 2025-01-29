import os

from starling.utilities import fix_ref_to_home

DEFAULT_MODEL_DIR = os.path.join("~/", ".starling_weights")
DEFAULT_ENCODE_WEIGHTS = "model-kernel-epoch=99-epoch_val_loss=1.72.ckpt"
DEFAULT_DDPM_WEIGHTS = "model-kernel-epoch=47-epoch_val_loss=0.03.ckpt"
DEFAULT_NUMBER_CONFS = 200
DEFAULT_BATCH_SIZE = 100
DEFAULT_STEPS = 10
DEFAULT_MDS_NUM_INIT = 4
DEFAULT_STRUCTURE_GEN = "mds"
CONVERT_ANGSTROM_TO_NM = 10
MAX_SEQUENCE_LENGTH = 384  # set longest sequence the model can work on


### Derived default values

# default paths to the model weights
DEFAULT_ENCODER_WEIGHTS_PATH = fix_ref_to_home(
    os.path.join(DEFAULT_MODEL_DIR, DEFAULT_ENCODE_WEIGHTS)
)
DEFAULT_DDPM_WEIGHTS_PATH = fix_ref_to_home(
    os.path.join(DEFAULT_MODEL_DIR, DEFAULT_DDPM_WEIGHTS)
)

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
