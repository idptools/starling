import argparse
import os

import yaml
from IPython import embed


def get_params(config_file: str = None) -> dict:
    """
    A function that reads the default configuration file
    and merges it with the user configuration file.

    Parameters
    ----------
    config_file : str, optional
        A path to the user configuration file, by default None

    Returns
    -------
    dict
        A dictionary containing the configuration parameters
    """
    dirname = os.path.dirname(__file__)
    with open(f"{dirname}/default_config.yaml", "r") as stream:
        default_config = yaml.safe_load(stream)

    if config_file is not None:
        with open(config_file, "r") as stream:
            user_config = yaml.safe_load(stream)
        # Merge user_config with default_config
    for key in user_config:
        default_config[key].update(user_config[key])

    return default_config
