

#
# Core utilities for the package. This should not importa anything from within starling
# to avoid circular imports.
#

import os
import torch

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
