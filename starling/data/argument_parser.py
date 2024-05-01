import argparse
import os

import yaml
from IPython import embed


def get_params(config_file=None):
    dirname = os.path.dirname(__file__)
    with open(f"{dirname}/default_config.yaml", "r") as stream:
        default_config = yaml.safe_load(stream)

    if config_file is not None:
        with open(config_file, "r") as stream:
            user_config = yaml.safe_load(stream)

        # Merge user_config with default_config
        config = {**default_config, **user_config}
    else:
        config = {**default_config}

    with open(f"{config['training']['output_path']}/config.yaml", "w") as f:
        yaml.dump(config, f)

    return config
