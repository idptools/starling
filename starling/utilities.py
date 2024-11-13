

#
# Core utilities for the package. This should not importa anything from within starling
# to avoid circular imports.
#

import os

def fix_ref_to_home(path):
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
    if path.startswith('~'):
        return os.path.expanduser(path)
    return path

