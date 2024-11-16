

#
# Core utilities for the package. This should not importa anything from within starling
# to avoid circular imports.
#

import os

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

