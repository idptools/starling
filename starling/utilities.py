

#
# Core utilities for the package. This should not importa anything from within starling
# to avoid circular imports.
#

import os
import torch
import pickle

def fix_ref_to_home(input_path):
    '''
    Function to fix the path to the home directory. 

    Parameters
    ---------------
    path : str
        The path to fix. 

    Returns
    ---------------
    str: The fixed path. 
    '''
    if input_path.startswith('~'):
        return os.path.expanduser(input_path)
    return input_path


def check_file_exists(input_path):
    '''
    Function to check if a file exists. 

    Parameters
    ---------------
    path : str
        The path to check. 

    Returns
    ---------------
    bool: True if the file exists, False otherwise. 
    '''
    return os.path.exists(input_path) and os.path.isfile(input_path)


def remove_extension(input_path):
    '''
    Function to remove the extension from a file.

    Parameters
    ---------------
    path : str
        The path to remove the extension from. 

    Returns
    ---------------
    str  
        The path with the extension removed. 

    '''

    return os.path.splitext(input_path)[0]


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


def check_device(use_device, default_device='gpu'):
    '''
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
    '''
    # Helper function to get CUDA device string (e.g., 'cuda:0', 'cuda:1')
    def get_cuda_device(cuda_str):
        if cuda_str == 'cuda':
            return torch.device('cuda')
        else:
            device_index = int(cuda_str.split(":")[1])
            num_devices = torch.cuda.device_count()
            if device_index >= num_devices:
                raise ValueError(f"{cuda_str} specified, but only {num_devices} CUDA-enabled GPUs are available. "
                                 f"Valid device indices are from 0 to {num_devices-1}.")
            return torch.device(cuda_str)

   # If `use_device` is None, fall back to `default_device`
    if use_device is None:
        default_device = default_device.lower()
        if default_device == 'cpu':
            return torch.device('cpu')
        elif default_device == 'gpu':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            raise ValueError("Default device can only be set to 'cpu' or 'gpu'.")
    
    # if a device is passed as torch.device, change to string so we
    # can make lowercase str for handling different device types. 
    if isinstance(use_device, torch.device):
        use_device = str(use_device)

    # Ensure `use_device` is a string
    if not isinstance(use_device, str):
        raise ValueError("Device must be type torch.device or string, valid options are: 'cpu', 'mps', 'cuda', or 'cuda:int'.")

    # make lower case to make checks easier. 
    use_device = use_device.lower()
    
    # Handle specific device strings
    if use_device == 'cpu':
        return torch.device('cpu')
    
    elif use_device == 'mps':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            raise ValueError("MPS was specified, but MPS is not available. Make sure you're running on an Apple device with MPS support.")

    elif use_device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError(f"{use_device} was specified, but torch.cuda.is_available() returned False.")
        return get_cuda_device(use_device)

    else:
        raise ValueError("Device must be 'cpu', 'mps', 'cuda', or 'cuda:int' (where int is a valid GPU index).")

    # This should never be reached
    raise RuntimeError("Unexpected state in the check_device function. Please raise an issue on GitHub.")


def write_starling_ensemble(ensemble_object, filename):
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

    
    """

    # build_the save dictionary
    save_dict = {'sequence': ensemble_object.sequence, 
                'distance_maps': ensemble_object._Ensemble__distance_maps, 
                'traj':ensemble_object._Ensemble__trajectory,
                'DEFAULT_ENCODER_WEIGHTS_PATH': ensemble_object._Ensemble__metadata['DEFAULT_ENCODER_WEIGHTS_PATH'],
                'DEFAULT_DDPM_WEIGHTS_PATH': ensemble_object._Ensemble__metadata['DEFAULT_DDPM_WEIGHTS_PATH'],
                'VERSION': ensemble_object._Ensemble__metadata['VERSION'],
                'DATE': ensemble_object._Ensemble__metadata['DATE']}
    
    # Remove the extension if it exists
    filename = remove_extension(filename)

    # add starling extension
    filename = filename + '.starling'

    # Save the dictionary to a file
    with open(filename, 'wb') as file:
        pickle.dump(save_dict, file)


def read_starling_ensemble(filename):        
    """
    Function to read a STARLING ensemble from a file in the STARLING
    format (.starling). This is actially just a dictionary with the 
    amino acid sequence, the distance maps, and the SSProtein object
    if available.

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

    # Read the dictionary from the file
    try:
        with open(filename, 'rb') as file:
            return_dict = pickle.load(file)
    except Exception:
        raise ValueError(f"Could not read the file {filename}. Please check the path and try again.")
    
    return return_dict
    

    

