import os

import numpy as np
import protfasta

from starling import configs, utilities
from starling.inference import generation


def handle_input(
    user_input, invalid_sequence_action="convert", output_name=None, seq_index_start=1
):
    """
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
    """

    # Helper function to validate and clean sequences.
    # This will raise an Exception if the sequence contains non-valid amino acids.
    # and makes sure everything is uppercase.
    def clean_sequence(sequence):
        sequence = sequence.upper()
        valid_residues = set("ACDEFGHIKLMNPQRSTVWY")  # Standard amino acids
        cleaned = "".join([res for res in sequence if res in valid_residues])
        # check lengths
        if len(cleaned) != len(sequence):
            raise ValueError(f"Invalid amino acid detected in sequence: {sequence}")
        # return the cleaned sequence.
        return cleaned

    ### Check and handle different input types

    # If input is a string...
    if isinstance(user_input, str):
        if user_input.endswith((".fasta", ".FASTA", ".tsv", ".in")):
            # make sure user has a valid path.
            if not os.path.exists(user_input):
                raise FileNotFoundError(f"File {user_input} not found.")
            # if a .fasta, use protfasta.
            if user_input.endswith((".fasta", ".FASTA")):
                # this will throw an error if we have duplicate sequence names, so
                # don't need to worry about that here.
                sequence_dict = protfasta.read_fasta(
                    user_input, invalid_sequence_action=invalid_sequence_action
                )
            elif user_input.endswith((".tsv", ".in")):
                # this doesn't have a check for duplicate sequences names,
                # so we should do that here. This should be the only instance
                # where that might cause an issue. Current behavior is to force duplicate names.
                sequence_dict = {}
                with open(user_input, "r") as f:
                    for line in f:
                        name, seq = line.strip().split("\t")
                        if name not in sequence_dict:
                            sequence_dict[name] = clean_sequence(seq)
                        else:
                            raise ValueError(
                                f"Duplicate sequence name detected: {name}.\nPlease ensure sequences have unique names in the input file."
                            )
            return sequence_dict
        else:
            # otherwise only string input allowed is a sequence as a string. Automatically create
            # the name if output_name is None.
            if output_name is None:
                return {f"sequence_{seq_index_start}": clean_sequence(user_input)}

            # if output_name is not None, use that as the key in the dictionary.
            else:
                try:
                    output_name = str(output_name)
                except Exception as e:
                    raise ValueError(
                        "output_name must be a string our castable to a string."
                    )
                return {output_name: clean_sequence(user_input)}

    # if input is a list
    elif isinstance(user_input, list):
        # if a list, make sure all sequences are valid.
        sequence_dict = {}
        for i, seq in enumerate(user_input):
            sequence_dict[f"sequence_{i + seq_index_start}"] = clean_sequence(seq)
        return sequence_dict
    elif isinstance(user_input, dict):
        # if a dict, make sure all sequences are valid.
        sequence_dict = {}
        for name, seq in user_input.items():
            sequence_dict[name] = clean_sequence(seq)
        return sequence_dict
    else:
        raise ValueError(
            f"Invalid input type: {type(user_input)}. Must be str, list, or dict."
        )


def check_positive_int(val):
    """
    Function to check if a value is a positive integer.

    Parameters
    ---------------
    val : int
        The value to check.

    Returns
    ---------------
    bool: True if val is a positive integer, False otherwise.
    """
    if isinstance(val, int) or np.issubdtype(type(val), np.integer):
        if val > 0:
            return True
    return False


def generate(
    user_input,
    conformations=configs.DEFAULT_NUMBER_CONFS,
    ionic_strength=configs.DEFAULT_IONIC_STRENGTH,
    device=None,
    steps=configs.DEFAULT_STEPS,
    sampler=configs.DEFAULT_SAMPLER,
    return_structures=False,
    batch_size=configs.DEFAULT_BATCH_SIZE,
    num_cpus_mds=configs.DEFAULT_CPU_COUNT_MDS,
    num_mds_init=configs.DEFAULT_MDS_NUM_INIT,
    output_directory=None,
    output_name=None,
    return_data=True,
    verbose=False,
    show_progress_bar=True,
    show_per_step_progress_bar=True,
    pdb_trajectory=False,
    return_single_ensemble=False,
    constraint=None,
    encoder_path=None,
    ddpm_path=None,
):
    """
    Generate STARLING ensembles and distance maps for one or more sequences.

    This is the primary high-level interface for STARLING ensemble generation. It
    normalizes the provided sequences, runs the diffusion sampler, optionally
    performs MDS refinement, and returns ensemble objects or writes them to disk.

    Parameters
    ----------
    user_input : str or Sequence[str] or Mapping[str, str]
        Input sequences to process. Supported forms include:

        * Path to a FASTA, TSV, or ``seq.in`` file containing name/sequence rows.
        * Raw amino-acid sequence string.
        * Iterable of sequence strings.
        * Mapping of sequence names to amino-acid sequences.

        Non-canonical residues trigger a :class:`ValueError`.
    conformations : int, default=configs.DEFAULT_NUMBER_CONFS
        Number of conformations to sample per sequence.
    ionic_strength : float, default=configs.DEFAULT_IONIC_STRENGTH
        Ionic strength (mM) supplied to the generative model.
    device : str or None, default=None
        Device identifier (``'cuda'``, ``'mps'``, or ``'cpu'``). ``None`` selects the
        best available accelerator.
    steps : int, default=configs.DEFAULT_STEPS
        Number of denoising diffusion steps.
    sampler : str, default=configs.DEFAULT_SAMPLER
        Sampler backend registered in :mod:`starling.configs`.
    return_structures : bool, default=False
        When ``True`` include 3D coordinate ensembles in the results.
    batch_size : int, default=configs.DEFAULT_BATCH_SIZE
        Batch size used for sampling iterations.
    num_cpus_mds : int, default=configs.DEFAULT_CPU_COUNT_MDS
        Number of CPU workers allocated to the MDS refinement stage.
    num_mds_init : int, default=configs.DEFAULT_MDS_NUM_INIT
        Number of independent MDS initializations to run per sequence.
    output_directory : str or os.PathLike or None, default=None
        Directory where generated outputs are written. When ``None`` nothing is saved.
    output_name : str or None, default=None
        Override the generated sequence key when a single sequence string is provided.
    return_data : bool, default=True
        When ``True`` return ensembles; otherwise the function returns ``None``.
    verbose : bool, default=False
        Emit status messages during generation.
    show_progress_bar : bool, default=True
        Display a global diffusion progress bar.
    show_per_step_progress_bar : bool, default=True
        Display an inner progress bar for per-step diffusion updates.
    pdb_trajectory : bool, default=False
        When ``True`` write PDB trajectories alongside XTC files. Only applies when
        ``return_structures`` is ``True`` or an ``output_directory`` is provided.
    return_single_ensemble : bool, default=False
        When ``True`` and exactly one sequence is processed, return a single
        :class:`starling.structure.ensemble.Ensemble`. Raises :class:`ValueError`
        if multiple sequences are supplied.
    constraint : Optional[starling.inference.constraints.Constraint], default=None
        Constraint object applied during sampling.
    encoder_path : str or os.PathLike or None, default=None
        Custom encoder checkpoint path overriding the configured default.
    ddpm_path : str or os.PathLike or None, default=None
        Custom diffusion model checkpoint path overriding the configured default.

    Returns
    -------
    dict[str, starling.structure.ensemble.Ensemble] or starling.structure.ensemble.Ensemble or None
        Dictionary of ensembles keyed by sequence name when ``return_data`` is ``True``.
        A single ensemble object is returned when ``return_single_ensemble`` is ``True``.
        Returns ``None`` when ``return_data`` is ``False``.

    Raises
    ------
    FileNotFoundError
        If the input path or output directory cannot be located.
    ValueError
        If sequences contain non-canonical residues or argument combinations are invalid.

    """
    # check user input, return a sequence dict.
    _sequence_dict = handle_input(user_input, output_name=output_name)

    # we do this specific sanity check EARLY so we don't silently fix what would
    # otherwise be a faulty input
    if return_single_ensemble and len(_sequence_dict) > 1:
        raise ValueError(
            f"Error: requested single ensemble yet provided input of {len(_sequence_dict)} sequences."
        )

    # filter out sequences that are too long (rather than erroring out)
    sequence_dict = {}
    removed_counter = 0
    for k in _sequence_dict:
        if len(_sequence_dict[k]) > configs.MAX_SEQUENCE_LENGTH:
            print(
                f"Warning: Sequence {k} is too long; maximum sequence in STARLING is {configs.MAX_SEQUENCE_LENGTH} residues, {k} is {len(_sequence_dict[k])}. Skipping..."
            )
            removed_counter = removed_counter + 1
        else:
            sequence_dict[k] = _sequence_dict[k]

    if verbose:
        # if we removed one sequence for being too long...
        if removed_counter == 1:
            bonus_message = f". Removed {removed_counter} sequence for being too long"

        # if we removed more than one sequence for being too long....
        elif removed_counter > 1:
            bonus_message = f". Removed {removed_counter} sequences for being too long"

        # if we removed no sequences!
        else:
            bonus_message = ""

        if len(sequence_dict) == 1:
            print(f"[STATUS]: Generating distance maps for 1 sequence{bonus_message}.")
        else:
            print(
                f"[STATUS]: Generating distance maps for {len(sequence_dict)} sequences{bonus_message}."
            )

    # check various other things so we fail early. Don't
    # want to go about the entire process and then have it fail at the end.
    # check conformations
    if not check_positive_int(conformations):
        raise ValueError("Error: Conformations must be an integer greater than 0.")

    # check steps
    if not check_positive_int(steps):
        raise ValueError("Error: Steps must be an integer greater than 0.")

    # check batch size
    if not check_positive_int(batch_size):
        raise ValueError("Error: batch_size must be an integer greater than 0.")

    # check number of cpus
    if not check_positive_int(num_cpus_mds):
        raise ValueError("Error: num_cpus_mds must be an integer greater than 0.")

    # check number of independent runs of MDS
    if not check_positive_int(num_mds_init):
        raise ValueError("Error: num_mds_init must be an integer greater than 0.")

    # make sure batch_size is not smaller than conformations.
    # if it is, make batch_size = conformations.
    if batch_size > conformations:
        batch_size = conformations

    # check output_directory is a directory that exists.
    if output_directory is not None:
        if not os.path.exists(output_directory):
            raise FileNotFoundError(
                f"Error: Directory {output_directory} does not exist."
            )

    # check sampler is a string
    if not isinstance(sampler, str):
        raise ValueError("Error: sampler must be a string.")

    # check return_structures is a bool
    if not isinstance(return_structures, bool):
        raise ValueError("Error: return_structures must be True or False.")

    # check verbose is a bool
    if not isinstance(verbose, bool):
        raise ValueError("Error: verbose must be True or False.")

    # check show_progress_bar
    if not isinstance(show_progress_bar, bool):
        raise ValueError("Error: show_progress_bar must be True or False.")

    # check show_per_step_progress_bar
    if not isinstance(show_per_step_progress_bar, bool):
        raise ValueError("Error: show_per_step_progress_bar must be True or False.")

    # we do this specific sanity check to make the logic later in this function easier
    if return_single_ensemble and return_data is False:
        raise ValueError(
            "Error: requested single ensemble yet also did not request data to be returned."
        )

    if return_data is False and output_directory is None:
        raise ValueError(
            "Error: both no return data (return_data=False) and also did not specifiy an output_directory; this means no output will be returned/saved anywhere, which is probably not desired!"
        )

    # check device, get back a torch.device (not a str!)
    device = utilities.check_device(device)

    # run the actual inference and return the results
    ensemble_return = generation.generate_backend(
        sequence_dict,
        conformations,
        device,
        steps,
        sampler,
        return_structures,
        batch_size,
        num_cpus_mds,
        num_mds_init,
        output_directory,
        return_data,
        verbose,
        show_progress_bar,
        show_per_step_progress_bar,
        pdb_trajectory,
        ionic_strength=ionic_strength,
        constraint=constraint,
        model_manager=generation.model_manager,
        encoder_path=encoder_path,
        ddpm_path=ddpm_path,
    )

    # if this is true we KNOW there is only one Ensemble in the return dict because
    # we previously checked for this.
    if return_single_ensemble:
        return list(ensemble_return.values())[0]
    else:
        return ensemble_return


def ensemble_encoder(
    ensemble,
    batch_size=32,
    device=None,
    output_directory=None,
    encoder_path=None,
    ddpm_path=None,
):
    # check device, get back a torch.device (not a str!)
    device = utilities.check_device(device)

    embeddings = generation.ensemble_encoder_backend(
        ensemble=ensemble,
        device=device,
        batch_size=batch_size,
        output_directory=output_directory,
        model_manager=generation.model_manager,
        encoder_path=encoder_path,
        ddpm_path=ddpm_path,
    )

    return embeddings


def sequence_encoder(
    sequence_dict,
    ionic_strength=configs.DEFAULT_IONIC_STRENGTH,
    batch_size=32,
    aggregate=False,
    device=None,
    output_directory=None,
    encoder_path=None,
    ddpm_path=None,
    pretokenized: bool = False,
    bucket: bool = False,
    bucket_size: int = 32,
    free_cuda_cache: bool = False,
    return_on_cpu: bool = True,
):
    """Embed sequences with the STARLING encoder.

    Parameters
    ----------
    sequence_dict : str | Sequence[str] | dict[str, str]
        Input sequences to encode. Accepts a FASTA/TSV path, a single sequence,
        a list of sequences, or a mapping of identifiers to sequences. The
        helper :func:`handle_input` normalizes the value into a
        ``{name: sequence}`` dictionary and validates residue alphabets.
    ionic_strength : int, optional
        Ionic strength (in mM) to condition the encoder. Valid values are
        typically 20, 150, or 300, matching the training regimes. Defaults to
        :data:`configs.DEFAULT_IONIC_STRENGTH`.
    batch_size : int, optional
        Number of sequences to process per batch.
    aggregate : bool, optional
        When ``True`` the function returns a single embedding vector per
        sequence using mean pooling. When ``False`` (default) residue-level
        embeddings are returned.
    device : str | torch.device | None, optional
        Device hint forwarded to :func:`utilities.check_device`. ``None`` lets
        STARLING pick the best available accelerator.
    output_directory : str | pathlib.Path | None, optional
        Directory for optional on-disk exports. Leave ``None`` to keep
        embeddings in memory only.
    encoder_path : str | None, optional
        Override the default encoder checkpoint.
    ddpm_path : str | None, optional
        Override the default diffusion checkpoint used by the shared model
        manager.
    pretokenized : bool, optional
        Set to ``True`` when ``sequence_dict`` already contains cached tokens to
        skip preprocessing.
    bucket : bool, optional
        Enable adaptive bucketing by sequence length to reduce padding waste in
        large batches.
    bucket_size : int, optional
        Maximum number of unique lengths per bucket when ``bucket`` is
        ``True``. Ignored otherwise.
    free_cuda_cache : bool, optional
        Release CUDA memory after each batch for long inference jobs.
    return_on_cpu : bool, optional
        Convert embeddings to CPU tensors before returning. Set to ``False`` to
        keep them on the selected device for downstream GPU workflows.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping from sequence identifiers to embedding tensors. The trailing
        tensor shape is ``(L, D)`` for residue-level embeddings or ``(D,)`` for
        aggregated embeddings, where ``L`` is sequence length and ``D`` is the
        latent dimension.

    Notes
    -----
    The encoder shares weights with the ensemble generator, so successive calls
    reuse cached models through ``generation.model_manager``. Use
    ``encoder_path`` and ``ddpm_path`` to experiment with fine-tuned weights
    without mutating global configuration.
    """
    # check device, get back a torch.device (not a str!)
    device = utilities.check_device(device)

    sequence_dict = handle_input(sequence_dict)

    embeddings = generation.sequence_encoder_backend(
        sequence_dict=sequence_dict,
        ionic_strength=ionic_strength,
        aggregate=aggregate,
        device=device,
        batch_size=batch_size,
        output_directory=output_directory,
        model_manager=generation.model_manager,
        encoder_path=encoder_path,
        ddpm_path=ddpm_path,
        pretokenized=pretokenized,
        bucket=bucket,
        bucket_size=bucket_size,
        free_cuda_cache=free_cuda_cache,
        return_on_cpu=return_on_cpu,
    )

    return embeddings
