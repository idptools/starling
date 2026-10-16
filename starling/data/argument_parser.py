import yaml


def get_params(config_file: str | None = None) -> dict:
    """
    A function that reads a YAML configuration file and returns its contents.
    No default configuration is merged in; the file must contain every parameter.

    Parameters
    ----------
    config_file : str
        A path to the YAML configuration file. Although the argument defaults to
        None, it is required.

    Returns
    -------
    dict
        A dictionary containing the configuration parameters

    Raises
    ------
    ValueError
        If config_file is None.
    """
    if config_file is None:
        raise ValueError("config_file is required")
    with open(config_file, "r") as stream:
        user_config = yaml.safe_load(stream)

    return user_config


def get_vae_params(config_file: str) -> dict:
    """
    A function that reads a YAML configuration file and returns its contents.
    No default configuration is merged in; the file must contain every parameter.

    Parameters
    ----------
    config_file : str
        A path to the YAML configuration file.

    Returns
    -------
    dict
        A dictionary containing the configuration parameters
    """

    with open(config_file, "r") as stream:
        user_config = yaml.safe_load(stream)

    return user_config
