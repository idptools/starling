#
# Core utilities for the package. This should not importa anything from within starling
# to avoid circular imports.
#

import gzip
import lzma
import os
import pickle
import warnings
import platform

import numpy as np
import torch

# code that allows access to the data directory
_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
    return os.path.join(_ROOT, "data", path)


def fix_ref_to_home(input_path):
    """
    Function to fix the path to the home directory.

    Parameters
    ---------------
    path : str
        The path to fix.

    Returns
    ---------------
    str: The fixed path.
    """
    if input_path.startswith("~"):
        return os.path.expanduser(input_path)
    return input_path


def check_file_exists(input_path):
    """
    Function to check if a file exists.

    Parameters
    ---------------
    path : str
        The path to check.

    Returns
    ---------------
    bool: True if the file exists, False otherwise.
    """
    return os.path.exists(input_path) and os.path.isfile(input_path)


def remove_extension(input_path, verbose=True):
    """
    Function to remove the extension from a file.

    Parameters
    ---------------
    path : str
        The path to remove the extension from.

    verbose : str
        Define how noisy to be...

    Returns
    ---------------
    str
        The path with the extension removed.

    """
    new_filename = os.path.splitext(input_path)[0]

    # added this in so we don't silently edit away a filename with a period
    # that would be invisible...
    if verbose:
        if input_path != new_filename:
            print(
                f"Warning: removed file extension from input file name.\nWas: {input_path}\nNow: {new_filename}"
            )

    return new_filename


def parse_output_path(args):
    """
    Parse the output path from the command line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.

    Returns
    -------
    str
        The output path and filename without an extension.
    """

    # get the filename (+extension) if the input file, without
    # any path info
    input_filename = os.path.basename(args.input_file)

    # if no output is specified, use the current directory;
    # this is the default behavior
    if args.output == ".":
        outname = input_filename

    # if input was provided for the output
    else:
        # if we were passed a path
        if os.path.isdir(args.output):
            outname = os.path.join(args.output, input_filename)
        else:
            outname = args.output
    # remove the extension
    outname = os.path.splitext(outname)[0]

    return outname

def get_macOS_version():
    """
    Function to check the macOS version.

    Returns
    ---------------
    int
        Returns -1 or the major macOS version.
    
    """
    # get the macOS version
    macos_version = platform.mac_ver()[0]

    if len(macos_version) == 0:
        return -1
    else:
        try:
            return int(macos_version.split('.')[0])
        except ValueError:
            return -1



def check_device(use_device, default_device="gpu"):
    """
    Function to check the device was correctly set.

    Parameters
    ---------------
    use_device : str
        Identifier for the device to be used for predictions.
        Possible inputs: 'cpu', 'mps', 'cuda', 'cuda:int', where the int corresponds to
        the index of a specific cuda-enabled GPU. If 'cuda' is specified and
        cuda.is_available() returns False, this will raise an Exception
        If 'mps' is specified and mps is not available, an exception will be raised.

    default_device : str
        The default device to use if device=None.
        If device=None and default_device != 'cpu' and default_device is
        not available, device_string will be returned as 'cpu'.
        Default is 'gpu'. This checks first for cuda and then for mps
        because STARLING is faster on both than it is on CPU, so we should
        use the fastest device available.
        Options are 'cpu' or 'gpu'

    Returns
    ---------------
    torch.device: A PyTorch device object representing the device to use.
    """

    # Helper function to get CUDA device string (e.g., 'cuda:0', 'cuda:1')
    def get_cuda_device(cuda_str):
        if cuda_str == "cuda":
            return torch.device("cuda")
        else:
            device_index = int(cuda_str.split(":")[1])
            num_devices = torch.cuda.device_count()
            if device_index >= num_devices:
                raise ValueError(
                    f"{cuda_str} specified, but only {num_devices} CUDA-enabled GPUs are available. "
                    f"Valid device indices are from 0 to {num_devices - 1}."
                )
            return torch.device(cuda_str)

    # If `use_device` is None, fall back to `default_device`
    if use_device is None:
        default_device = default_device.lower()
        if default_device == "cpu":
            return torch.device("cpu")
        elif default_device == "gpu":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            raise ValueError("Default device can only be set to 'cpu' or 'gpu'.")

    # if a device is passed as torch.device, change to string so we
    # can make lowercase str for handling different device types.
    if isinstance(use_device, torch.device):
        use_device = str(use_device)

    # Ensure `use_device` is a string
    if not isinstance(use_device, str):
        raise ValueError(
            "Device must be type torch.device or string, valid options are: 'cpu', 'mps', 'cuda', or 'cuda:int'."
        )

    # make lower case to make checks easier.
    use_device = use_device.lower()

    # Handle specific device strings
    if use_device == "cpu":
        return torch.device("cpu")

    elif use_device == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            raise ValueError(
                "MPS was specified, but MPS is not available. Make sure you're running on an Apple device with MPS support."
            )

    elif use_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                f"{use_device} was specified, but torch.cuda.is_available() returned False."
            )
        return get_cuda_device(use_device)

    else:
        raise ValueError(
            "Device must be 'cpu', 'mps', 'cuda', or 'cuda:int' (where int is a valid GPU index)."
        )

    # This should never be reached
    raise RuntimeError(
        "Unexpected state in the check_device function. Please raise an issue on GitHub."
    )


def write_starling_ensemble(
    ensemble_object,
    filename,
    compress=False,
    reduce_precision=None,
    compression_algorithm="lzma",
    verbose=True,
):
    """
    Function to write the STARLING ensemble to a file in the STARLING
    format (.starling). This is actially just a dictionary with the
    amino acid sequence, the distance maps, and the SSProtein object
    if available.

    Parameters
    ---------------
    ensemble_object : starling.structure.Ensemble
        The STARLING ensemble object to save to a file.

    filename : str
        The filename to save the ensemble to; note this should
        not include a file extenison and if it does this will
        be removed

    compress : bool
        Whether to compress the file or not. Default is False.

    reduce_precision : bool
        Whether to reduce the precision of the distance map to a
        single decimal point and cast to float16 if possible.
        Default is None. Sets to False if compression is False,
        but True if compression is True.

    compression_algorithm : str
        The compression algorithm to use. Options are 'gzip' and 'lzma'.
        `lzma` gives better compression if reduce_precision is set to True,
        but actually 'gzip' is better if reduce_precision is False. 'lzma'
        is also slower than 'gzip'. Default is 'lzma'.

    verbose : bool
        Flag to define how noisy we should be
    """

    # set reduce_precision to mirror compress if not set
    if reduce_precision is None:
        reduce_precision = compress

    # build_the save dictionary
    save_dict = {
        "sequence": ensemble_object.sequence,
        "distance_maps": ensemble_object._Ensemble__distance_maps,
        "traj": ensemble_object._Ensemble__trajectory,
        "DEFAULT_ENCODER_WEIGHTS_PATH": ensemble_object._Ensemble__metadata[
            "DEFAULT_ENCODER_WEIGHTS_PATH"
        ],
        "DEFAULT_DDPM_WEIGHTS_PATH": ensemble_object._Ensemble__metadata[
            "DEFAULT_DDPM_WEIGHTS_PATH"
        ],
        "VERSION": ensemble_object._Ensemble__metadata["VERSION"],
        "DATE": ensemble_object._Ensemble__metadata["DATE"],
    }

    # if we wish to reduce the precision of the distance map to a single decimal point
    if reduce_precision:
        # run here so we catch warnings if they happen
        with warnings.catch_warnings(record=True) as w:
            # cast to float16 array and round to 1 decimal place
            tmp = np.round(ensemble_object._Ensemble__distance_maps, decimals=1).astype(
                "float16"
            )

            # check if a RuntimeWarning was raised
            for warning in w:
                # IF yes, then we actually don't cast because we can't do so faithfully
                if issubclass(warning.category, RuntimeWarning):
                    print(
                        "Warning: Could not reduce precision of distance maps to float16. Saving as float64."
                    )
                    tmp = np.round(ensemble_object._Ensemble__distance_maps, decimals=1)

        # update the save dictionary one way or another.
        save_dict["distance_maps"] = tmp

    # Remove the extension if it exists
    filename = remove_extension(filename, verbose=verbose)

    # add starling extension
    filename = filename + ".starling"

    # If we want to compress the file
    if compress:
        if compression_algorithm == "gzip":
            with gzip.open(filename + ".gzip", "wb") as file:
                pickle.dump(save_dict, file)

        elif compression_algorithm == "lzma":
            with lzma.open(filename + ".xz", "wb") as file:
                pickle.dump(save_dict, file)
        else:
            raise ValueError(
                f"Compression algorithm {compression_algorithm} is not supported. Supported algorithms are 'gzip' and 'lzma'."
            )

    else:
        with open(filename, "wb") as file:
            pickle.dump(save_dict, file)


def read_starling_ensemble(filename):
    """
    Function to read a STARLING ensemble from a file in the STARLING
    format (.starling) or a compressed starling file (.gzip, .xz).
    The .starling file is actially just a dictionary with the
    amino acid sequence, the distance maps, some metadata and the
    SSProtein object if available.

    Note this determines the file type based on the extension and
    will raise an error if the file is not a .starling file, a
    a .starling.gzip or .starling.xz file.

    Parameters
    ---------------
    filename : str
        The filename to read the ensemble from; note this should
        not include a file extenison and if it does this will
        be removed

    Returns
    ---------------
    starling.structure.Ensemble: The STARLING ensemble object
    """

    if filename.endswith(".starling.gzip"):
        try:
            with gzip.open(filename, "rb") as file:
                return_dict = pickle.load(file)
        except Exception:
            raise ValueError(
                f"Could not read the file {filename}. Please check the path and try again."
            )

    elif filename.endswith(".starling.xz"):
        try:
            with lzma.open(filename, "rb") as file:
                return_dict = pickle.load(file)
        except Exception:
            raise ValueError(
                f"Could not read the file {filename}. Please check the path and try again."
            )

    elif filename.endswith(".starling"):
        try:
            with open(filename, "rb") as file:
                return_dict = pickle.load(file)
        except Exception:
            raise ValueError(
                f"Could not read the file {filename}. Please check the path and try again."
            )

    else:
        raise ValueError(
            f"File {filename} does not have the extension of .starling.gzip, .starling.xz or .starling."
        )

    # NOTE: We if the distance maps are float16, we cast them to float32
    if return_dict["distance_maps"].dtype == "float16":
        return_dict["distance_maps"] = return_dict["distance_maps"].astype("float32")

    return return_dict


def symmetrize_distance_maps(dist_maps):
    """
    Symmetrizes a stack of distance maps along an axis by reflecting the upper triangle
    onto the lower triangle and setting the diagonal values to zero.

    Parameters:
    ----------
    dist_maps : np.ndarray
        A 3D NumPy array of shape (N, M, M), where N is the number of distance maps.

    Returns:
    -------
    np.ndarray
        A symmetrized stack of distance maps.
    """
    # Ensure input is a 3D array
    assert dist_maps.ndim == 3, "Input must be a 3D array of shape (N, M, M)."
    assert dist_maps.shape[1] == dist_maps.shape[2], "Each distance map must be square."

    # Create masks for the upper triangle
    M = dist_maps.shape[1]
    mask_upper_triangle = np.triu_indices(M, k=1)

    # Reflect the upper triangle onto the lower triangle
    dist_maps[:, mask_upper_triangle[1], mask_upper_triangle[0]] = dist_maps[
        :, mask_upper_triangle[0], mask_upper_triangle[1]
    ]

    # Set diagonal values to zero
    np.einsum("nii->ni", dist_maps)[:] = 0  # Efficient diagonal zeroing

    return dist_maps


def get_off_diagonals(
    distance_map, min_separation=1, max_separation=4, return_mean=False
):
    """
    Function to calculate the  the off-diagonal elements of a matrix.

    This is useful for error checking as we can KNOW the max distance between
    any pair of residues base on the contour length of the protein, allowing
    us to identify conformations that are simply impossible.

    Parameters
    ---------------
    distance_map : np.ndarray
        The distance map to check for errors.

    min_separation : int
        The minimum sequence separation to check across.

    max_separation : int
        The maxiumum sequence separation to check across.

    Returns
    ---------------
    np.ndarray
        All off-diagonal elements of the distance map from
        min_separation to max_separation away from the true
        diagonal.

    """

    # check we have a square...
    if distance_map.shape[0] != distance_map.shape[1]:
        raise ValueError("Input matrix must be square.")

    # get sequence length
    n = distance_map.shape[0]

    # build a single list with all the diagonal vals
    values = []
    for d in range(min_separation, max_separation + 1):
        values.extend(distance_map.diagonal(d))  # Upper diagonals
        values.extend(distance_map.diagonal(-d))  # Lower diagonals

    if return_mean:
        return np.mean(values)
    else:
        return np.array(values)


def check_distance_map_for_error(distance_map, min_separation=1, max_separation=2):
    """
    Function to check the distance map for errors.

    Parameters
    ---------------
    distance_map : np.ndarray
        The distance map to check for errors.

    min_separation : int
        The minimum sequence separation to check across.

    max_separation : int
        The maxiumum sequence separation to check across.

    Returns
    ---------------
    bool
        Returns True if an error was detected,
        and False if not.

    """

    ij_abs = abs(max_separation - min_separation) + 1

    # we can KNOW the max distance any i-j residues are
    # nb 3.81 is Mpipi bond length, but we add a +1 Angstrom to EACH
    # bond as an error term; this likely makes the efficacy of this approach
    # VERY poor if you have large sequence separation, but it also minimizes
    # the risk of false positives.
    # changes should update this but we'll hardcode it for now...
    max_possible_dist = 4.5 * ij_abs

    # get biggest distance from the set
    max_ods = np.max(
        get_off_diagonals(
            distance_map, min_separation=min_separation, max_separation=max_separation
        )
    )
    if max_ods > max_possible_dist:
        return True
    else:
        return False
