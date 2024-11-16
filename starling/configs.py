import os

from starling.utilities import fix_ref_to_home

DEFAULT_MODEL_DIR      = os.path.join('~/', '.starling_weights')
DEFAULT_ENCODE_WEIGHTS = 'model-kernel-epoch=99-epoch_val_loss=1.72.ckpt'
DEFAULT_DDPM_WEIGHTS   = 'model-kernel-epoch=08-epoch_val_loss=0.03.ckpt'
DEFAULT_NUMBER_CONFS   = 200
DEFAULT_BATCH_SIZE     = 100
DEFAULT_STEPS          = 10
DEFAULT_MDS_NUM_INIT   = 4
DEFAULT_STRUCTURE_GEN  = 'mds'





### Derived default values 

# default paths to the model weights
DEFAULT_ENCODER_WEIGHTS_PATH = fix_ref_to_home(os.path.join(DEFAULT_MODEL_DIR, DEFAULT_ENCODE_WEIGHTS))
DEFAULT_DDPM_WEIGHTS_PATH    = fix_ref_to_home(os.path.join(DEFAULT_MODEL_DIR, DEFAULT_DDPM_WEIGHTS))

# Set the default number of CPUs to use
DEFAULT_CPU_COUNT_MDS = min(DEFAULT_MDS_NUM_INIT, os.cpu_count())




