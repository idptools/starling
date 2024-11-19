import protfasta
import os

from tqdm import tqdm
import numpy as np
from scipy.spatial import distance_matrix


#
from starling import configs
from starling import utilities

from starling.inference import generation


def handle_input(user_input, 
                 invalid_sequence_action='convert',
                 output_name=None,
                 seq_index_start=1):
    '''
    Dynamically handle the input from the user.
    This returns a dictionary with either the names from 
    the user's input file or the users input dictionary of
    sequences or will create a dictionary with the sequences
    numbered in the order they were passed in with seq_index_start
    as the starting index. 

    Parameters
    -----------
    user_input: str, list, dict
        This can be one of a few different options:
            str: A .fasta file  
            str: A seq.in file formatted as a .tsv with name\tseq
            str: A .tsv file formatted as name\tseq. Same as seq.in
                except a different file extension. Borna used a seq.in
                in his tutorial, so I'm rolling with it. 
            str: A sequence as a string
            list: A list of sequences
            dict: A dict of sequences (name: seq)

    invalid_sequence_action: str
        This can be one of 3 options:
            fail - invalid sequence cause parsing to fail and throw an exception
            remove - invalid sequences are removed
            convert - invalid sequences are converted
            Default is 'convert'
            Only these 3 options are allowed because STARLING cannot handle
            non-canonical residues, so we don't want to use the protfasta.read_fasta()
            options that allow this to happen. 

    output_name : str
        If provided and if a single amino acid sequence is passed in, this will be the key
        in the output dictionary. If None, the key will be 'sequence_<index>'. If a dictionary
        or list or path to a FASTA file is passed, this is ignored. Default is None.
        
    seq_index_start: int
        If we need to number sequences in the output dictionary, this is the starting index.
        This is only needed if a sequence as a string is passed in or if a list of sequences
        is passed in. 

    Returns
    --------
    dict: A dictionary of sequences (name: seq)
    '''

    # Helper function to validate and clean sequences.
    # This will raise an Exception if the sequence contains non-valid amino acids.
    # and makes sure everything is uppercase.  
    def clean_sequence(sequence):
        sequence=sequence.upper()
        valid_residues = set("ACDEFGHIKLMNPQRSTVWY")  # Standard amino acids
        cleaned = ''.join([res for res in sequence if res in valid_residues])
        # check lengths
        if len(cleaned) != len(sequence):
            raise ValueError(f"Invalid amino acid detected in sequence: {sequence}")
        # return the cleaned sequence. 
        return cleaned

    ### Check and handle different input types

    # If input is a string...
    if isinstance(user_input, str):
        if user_input.endswith(('.fasta', '.FASTA', '.tsv', '.in')):
            # make sure user has a valid path.
            if not os.path.exists(user_input):
                raise FileNotFoundError(f"File {user_input} not found.")
            # if a .fasta, use protfasta.
            if user_input.endswith(('.fasta', '.FASTA')):
                # this will throw an error if we have duplicate sequence names, so 
                # don't need to worry about that here.
                sequence_dict=protfasta.read_fasta(user_input, 
                                                    invalid_sequence_action=invalid_sequence_action)
            elif user_input.endswith(('.tsv', '.in')):
                # this doesn't have a check for duplicate sequences names, 
                # so we should do that here. This should be the only instance
                # where that might cause an issue. Current behavior is to force duplicate names.
                sequence_dict={}
                with open(user_input, 'r') as f:
                    for line in f:
                        name, seq = line.strip().split('\t')
                        if name not in sequence_dict:
                            sequence_dict[name] = clean_sequence(seq)
                        else:
                            raise ValueError(f"Duplicate sequence name detected: {name}.\nPlease ensure sequences have unique names in the input file.")
            return sequence_dict
        else:

            # otherwise only string input allowed is a sequence as a string. Automatically create
            # the name if output_name is None.
            if output_name is None:                
                return {f'sequence_{seq_index_start}': clean_sequence(user_input)}
            
            # if output_name is not None, use that as the key in the dictionary.
            else:
                try:
                    output_name = str(output_name)
                except Exception as e:
                    raise ValueError('output_name must be a string our castable to a string.')
                return {output_name: clean_sequence(user_input)}
            
    # if input is a list
    elif isinstance(user_input, list):
        # if a list, make sure all sequences are valid.
        sequence_dict = {}
        for i, seq in enumerate(user_input):
            sequence_dict[f'sequence_{i+seq_index_start}'] = clean_sequence(seq)
        return sequence_dict
    elif isinstance(user_input, dict):
        # if a dict, make sure all sequences are valid.
        sequence_dict = {}
        for name, seq in user_input.items():
            sequence_dict[name] = clean_sequence(seq)
        return sequence_dict
    else:
        raise ValueError(f"Invalid input type: {type(user_input)}. Must be str, list, or dict.")




def check_positive_int(val):
    '''
    Function to check if a value is a positive integer. 

    Parameters
    ---------------
    val : int
        The value to check. 

    Returns
    ---------------
    bool: True if val is a positive integer, False otherwise.
    '''
    if isinstance(val, int):
        if val > 0:
            return True
    return False



def generate(user_input,
             conformations=configs.DEFAULT_NUMBER_CONFS, 
             device=None,
             steps=configs.DEFAULT_STEPS,
             method='mds',
             ddim=True,
             return_structures=False,
             batch_size=configs.DEFAULT_BATCH_SIZE, 
             num_cpus_mds=configs.DEFAULT_CPU_COUNT_MDS,
             num_mds_init=configs.DEFAULT_MDS_NUM_INIT,
             output_directory=None,
             output_name=None,
             return_data=True,
             verbose=False,
             show_progress_bar=True,
             show_per_step_progress_bar=True):
    '''
    Main function for generating the distance maps using STARLING. Allows
    you to pass a single sequence, a list of sequences, a dictionary, or
    a path to a .fasta file, a .tsv file, or a seq.in file, and from that
    generate distance maps and 3D conformational ensembles using the 
    STARLING model. This function is the main user-facing STARLING function.
    
    Note: if you want to change the location of the networks, 
    you need to change them in the configs.py file. Those paths get
    read in by the ModelManager class and are not passed in as arguments
    to this function. This lets us avoid iteratively loading the network
    when running the generate function multiple times in a single session.

    Parameters
    ---------------
    user_input : str, list, dict
        This can be one of a few different options:
            str: A path to a .fasta file as a str.
            str: A path to a seq.in file formatted as a .tsv with name\tseq
            str: A path to a .tsv file formatted as name\tseq. Same as
                seq.in except a different file extension. Borna used a seq.in
                in his tutorial, so I'm rolling with it.
            str: A sequence as a string
            list: A list of sequences
            dict: A dict of sequences (name: seq)

    conformations : int
        The number of conformations to generate. Default is 200.

    device : str
        The device to use for predictions. Default is None. If None, the
        default device will be used. Default device is 'gpu'.
        This is MPS for apple silicon and CUDA for all other devices.
        If MPS and CUDA are not available, automatically falls back to CPU.

    steps : int
        The number of steps to run the DDPM model. Default is 10.

    method : str
        The method to use for generating the 3D structure. Default is 'mds'.
    
    ddim : bool
        Whether to use DDIM for sampling. Default is True.

    return_structures : bool
        Whether to return the 3D structure. Default is False.

    batch_size : int
        The batch size to use for sampling. 100 uses ~20 GB 
        of memory. Default is 100.

    num_cpus_mds : int
        The number of CPUs to use for MDS. Default is 4

    num_mds_init : int
        Number of independent MDS jobs to execute. Default is 
        4. Note if goes up in principle more shots of finding
        a good solution but there is a performance hit unless
        num_cpus_mds == num_mds_init.

    output_directory : str
        The path to save the output.

        If set to None, no output will be saved to disk. 

        If not None, will save the output to the specified path.
        This includes the distance maps and, if 
        return_structures=True, the 3D structures.
        
        The distance maps are saved as .npy files with the names
        <sequence_name>_STARLING_DM.npy and the structures are 
        saved with the file names <sequence_name>_STARLING.xtc 
        and <sequence_name>_STARLING.pdb.

        <sequence_name> here will depend on the input provided
        to generate. If the input is a dictionary, then the keys
        will be used as the sequence names. If the input is a
        list or a single sequence, the sequence will be saved
        as 'sequence_<index>'. If a path to a FASTA file is passed
        in, the headers from the FASTA file will be used. Note also
        if a single sequence is passed the sequence_<index> format
        can be overridden by setting the output_name parameter.

        Default is None.

    output_name : str
        If provided and if a single amino acid sequence is passed in, 
        this will be the key in the output dictionary. If None, the 
        key will be 'sequence_<index>'. If a dictionary or list or path 
        to a FASTA file is passed, this is ignored. Default is None.

    return_data : bool
        If True, will return the distance maps and structures (if generated)
        as a dictionary. If False, will return None (so you need to set the
        output_dictionary, or the analysis will be lost!). Default is True

    verbose : bool
        Whether to print verbose output. Default is False.

    show_progress_bar : bool
        Whether to show a progress bar. Default is True.

    show_per_step_progress_bar : bool, optional
        Whether to show progress bar per step. 
        Default is True

    Returns
    ---------------
    dict or None: 
        A dict with the sequence names as the key and 
        an np.ndarray of the distance maps as the value. 

        If return_structures=True, the dict will a key that is
        the sequence name + '_traj' and the value will be the 
        structure as a mdtraj.Trajectory object. 

        If output_directory is not none, the output will save to 
        the specified path.
    '''
    # check user input, return a sequence dict. 
    sequence_dict = handle_input(user_input, output_name=output_name)

    if verbose:
        if len(sequence_dict) == 1:
            print(f"[STATUS]: Generating distance maps for 1 sequence.")
        else:
            print(f"[STATUS]: Generating distance maps for {len(sequence_dict)} sequences.")

    # check various other things so we fail early. Don't
    # want to go about the entire process and then have it fail at the end.
    # check conformations
    if not check_positive_int(conformations):
        raise ValueError("Conformations must be a positive integer.")
    
    # check steps
    if not check_positive_int(steps):
            raise ValueError("Steps must be an integer greater than 0.")
    
    # check batch size
    if not check_positive_int(batch_size):
        raise ValueError("batch_size must be an integer greater than 0.")
    
    # check number of cpus
    if not check_positive_int(num_cpus_mds):
        raise ValueError("num_cpus_mds must be an integer greater than 0.")
    
    # check number of independent runs of MDS
    if not check_positive_int(num_mds_init):
        raise ValueError("num_mds_init must be an integer greater than 0.")
        
    # make sure batch_size is not smaller than conformations.
    # if it is, make batch_size = conformations. 
    if batch_size > conformations:
        batch_size=conformations

    # make method lowercase and then check method
    method=method.lower()
    if method not in ['mds', 'gd']:
        raise ValueError("Method must be 'mds' or 'gd'.")

    # check output_directory is a directory that exists.
    if output_directory is not None:
        if not os.path.exists(output_directory):
            raise FileNotFoundError(f"Directory {output_directory} does not exist.")

    # check ddim is a bool
    if not isinstance(ddim, bool):
        raise ValueError("DDIM must True or False.")

    # check return_structures is a bool
    if not isinstance(return_structures, bool):
        raise ValueError("return_structures must be True or False.")

    # check verbose is a bool
    if not isinstance(verbose, bool):
        raise ValueError("verbose must be True or False.")

    # check show_progress_bar
    if not isinstance(show_progress_bar, bool):
        raise ValueError("show_progress_bar must be True or False.")

    # check show_per_step_progress_bar
    if not isinstance(show_per_step_progress_bar, bool):
        raise ValueError("show_per_step_progress_bar must be True or False.")

    # check device, get back torch.device
    device = utilities.check_device(device)

    # run the actual inference and return the results
    return generation.generate_backend(sequence_dict,
                                       conformations,
                                       device,
                                       steps,
                                       method,
                                       ddim,
                                       return_structures,
                                       batch_size,
                                       num_cpus_mds,
                                       num_mds_init,
                                       output_directory,
                                       return_data,
                                       verbose,
                                       show_progress_bar,
                                       show_per_step_progress_bar)

