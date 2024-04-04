import argparse
import os

import yaml
from IPython import embed


def get_params(config_file=None):
    default_config = {
        "model": {
            "in_channels": 1,
            "model": "ResNet18",
            "latent_dim": 128,
            "kernel_size": 3,
            "loss_type": "elbo",
            "weights_type": "reciprocal",
            "KLD_weight": 1,
            "lr_scheduler": "OneCycleLR",
            "dimension": 384,
        },
        "training": {
            "project_name": "testing_elbo",
            "learning_rate": None,
            "num_epochs": 100,
            "output_path": ".",
        },
        "data": {
            "train_data": None,
            "val_data": None,
            "test_data": None,
            "normalize": None,
            "batch_size": 64,
        },
        "device": {"cuda": [0]},
    }

    if config_file is not None:
        with open(config_file, "r") as stream:
            user_config = yaml.safe_load(stream)

        # Merge user_config with default_config
        config = {**default_config, **user_config}
    else:
        config = {**default_config}

    os.makedirs(config["training"]["output_path"], exist_ok=True)

    with open(f"{config['training']['output_path']}/config.yaml", "w") as f:
        yaml.dump(config, f)

    return config
