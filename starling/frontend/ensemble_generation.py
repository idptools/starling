import numbers
import os
import warnings

import numpy as np
import protfasta

from starling import configs, utilities
from starling.inference import generation
from starling.inference.constraints import Constraint, MultiConstraint
from starling.samplers.plms_sampler import PLMS_MIN_STEPS

# File extensions handle_input() reads as sequence files rather than treating
# the string as a sequence
INPUT_FILE_EXTENSIONS = (".fasta", ".FASTA", ".tsv", ".in")

# Seeds must be below this: torch.Generator.manual_seed takes 64-bit seeds
MAX_SEED_EXCLUSIVE = 2**64


def _report_fasta_changes(fasta_path, cleaned_sequences, invalid_sequence_action):
    """
    Warn about every sequence protfasta changed or removed while reading a FASTA file.

    protfasta converts or removes non-canonical residues without saying so,
    so the file is read a second time with invalid residues left in place and
    compared against the cleaned sequences.

    Parameters
    ----------
    fasta_path : str
        Path to the FASTA file.
    cleaned_sequences : dict
        Sequences (name: seq) as returned by protfasta.read_fasta() with
        ``invalid_sequence_action``.
    invalid_sequence_action : str
        The action used to clean the sequences (see :func:`handle_input`).

    Returns
    -------
    None
        One warning is issued per sequence that was changed or removed.
    """
    raw_sequences = protfasta.read_fasta(fasta_path, invalid_sequence_action="ignore")
    for name, raw_sequence in raw_sequences.items():
        if name not in cleaned_sequences:
            warnings.warn(
                f"Sequence {name} in {fasta_path} contains non-canonical residues and was removed "
                f"(invalid_sequence_action='{invalid_sequence_action}').",
                stacklevel=3,
            )
            continue

        cleaned_sequence = cleaned_sequences[name]
        if cleaned_sequence == raw_sequence:
            continue

        # protfasta's conversions are one residue for one residue, and its
        # removals ('*', '-', ' ') drop the character, so the length change
        # counts removals and the rest of the non-canonical residues were converted
        n_removed = len(raw_sequence) - len(cleaned_sequence)
        n_non_canonical = sum(residue not in configs.VALID_AA for residue in raw_sequence)
        n_converted = n_non_canonical - n_removed
        warnings.warn(
            f"Sequence {name} in {fasta_path} contains {n_non_canonical} non-canonical residue(s): "
            f"{n_converted} converted (X->G, B->N, U->C, Z->Q) and {n_removed} removed ('*', '-', ' ').",
            stacklevel=3,
        )


def handle_input(user_input, invalid_sequence_action="convert", output_name=None, seq_index_start=1):
    """
    Dynamically handle the input from the user.
    This returns a dictionary with either the names from
    the user's input file or the users input dictionary of
    sequences or will create a dictionary with the sequences
    numbered in the order they were passed in with seq_index_start
    as the starting index.

    Parameters
    ----------
    user_input : str, os.PathLike, list, tuple, or dict
        This can be one of a few different options:

        - str or os.PathLike: A .fasta file
        - str or os.PathLike: A seq.in file formatted as a .tsv with name\\tseq
        - str or os.PathLike: A .tsv file formatted as name\\tseq. Same as seq.in
          except a different file extension. Borna used a seq.in
          in his tutorial, so I'm rolling with it. Blank lines are skipped.
        - str: A sequence as a string
        - list or tuple: A list of sequences
        - dict: A dict of sequences (name: seq)

        Every sequence must contain at least one residue. Non-canonical
        residues raise a ValueError, except in FASTA files, where they are
        handled according to ``invalid_sequence_action``.

    invalid_sequence_action : str
        How non-canonical residues in a FASTA file are handled. This can be
        one of 3 options:

        - fail - invalid sequence cause parsing to fail and throw an exception
        - remove - invalid sequences are removed, with a warning naming each one
        - convert - invalid residues are converted (X->G, B->N, U->C, Z->Q)
          and '*', '-' and ' ' are removed, with a warning naming each
          sequence changed and how many residues were converted/removed

        Default is 'convert'.
        Only these 3 options are allowed because STARLING cannot handle
        non-canonical residues, so we don't want to use the protfasta.read_fasta()
        options that allow this to happen.

    output_name : str
        If provided and if a single amino acid sequence is passed in, this will be the key
        in the output dictionary. If None, the key will be 'sequence_<index>'. If a dictionary
        or list or path to a FASTA file is passed, this is ignored. Default is None.

    seq_index_start : int
        If we need to number sequences in the output dictionary, this is the starting index.
        This is only needed if a sequence as a string is passed in or if a list of sequences
        is passed in.

    Returns
    -------
    dict
        A dictionary of sequences (name: seq)

    Raises
    ------
    ValueError
        If a sequence is empty or contains non-canonical residues (outside
        FASTA files), a .tsv/.in line is not name<TAB>sequence, a path object
        does not have a supported extension, or the input type is unsupported.
    FileNotFoundError
        If an input file does not exist.
    """

    # Helper function to validate and clean sequences.
    # This will raise an Exception if the sequence is empty or contains non-valid
    # amino acids, and makes sure everything is uppercase.
    def clean_sequence(sequence):
        if len(sequence) == 0:
            raise ValueError("Empty sequence provided; every sequence must contain at least one residue.")
        sequence = sequence.upper()
        valid_residues = set("ACDEFGHIKLMNPQRSTVWY")  # Standard amino acids
        cleaned = "".join([res for res in sequence if res in valid_residues])
        # check lengths
        if len(cleaned) != len(sequence):
            raise ValueError(f"Invalid amino acid detected in sequence: {sequence}")
        # return the cleaned sequence.
        return cleaned

    ### Check and handle different input types

    # a path object (e.g. pathlib.Path) can only be a sequence file, never a
    # sequence, so convert it to a string path and insist on a known extension
    if isinstance(user_input, os.PathLike):
        user_input = os.fsdecode(user_input)
        if not user_input.endswith(INPUT_FILE_EXTENSIONS):
            raise ValueError(
                f"Input file {user_input} must have one of the extensions {', '.join(INPUT_FILE_EXTENSIONS)}."
            )

    # If input is a string...
    if isinstance(user_input, str):
        if user_input.endswith(INPUT_FILE_EXTENSIONS):
            # make sure user has a valid path.
            if not os.path.exists(user_input):
                raise FileNotFoundError(f"File {user_input} not found.")
            # if a .fasta, use protfasta.
            if user_input.endswith((".fasta", ".FASTA")):
                # this will throw an error if we have duplicate sequence names, so
                # don't need to worry about that here.
                sequence_dict = protfasta.read_fasta(user_input, invalid_sequence_action=invalid_sequence_action)
                _report_fasta_changes(user_input, sequence_dict, invalid_sequence_action)

                # a record holding only removable characters (e.g. '*') is empty once converted
                for name, seq in sequence_dict.items():
                    if len(seq) == 0:
                        raise ValueError(f"Sequence {name} in {user_input} is empty after removing invalid residues.")
            elif user_input.endswith((".tsv", ".in")):
                # this doesn't have a check for duplicate sequences names,
                # so we should do that here. This should be the only instance
                # where that might cause an issue. Current behavior is to force duplicate names.
                sequence_dict = {}
                with open(user_input, "r") as f:
                    for line_number, line in enumerate(f, start=1):
                        # skip blank lines, such as a trailing empty line at the end of the file
                        if not line.strip():
                            continue
                        fields = line.strip().split("\t")
                        if len(fields) != 2:
                            raise ValueError(
                                f"Line {line_number} of {user_input} is not formatted as name<TAB>sequence: "
                                f"{line.strip()!r}"
                            )
                        name, seq = fields
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
                except Exception:
                    raise ValueError("output_name must be a string our castable to a string.")
                return {output_name: clean_sequence(user_input)}

    # if input is a list (or tuple)
    elif isinstance(user_input, (list, tuple)):
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
        raise ValueError(f"Invalid input type: {type(user_input)}. Must be str, os.PathLike, list, tuple, or dict.")


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


def _prepare_constraint(constraint):
    """
    Check the constraint passed to generate(), combining a list of them into one.

    Parameters
    ----------
    constraint : Constraint, list or tuple of Constraint, or None
        The constraint(s) to guide sampling with.

    Returns
    -------
    Constraint or None
        ``constraint`` itself if it is a single Constraint or None, or a
        :class:`~starling.inference.constraints.MultiConstraint` combining a
        list or tuple of Constraints.

    Raises
    ------
    TypeError
        If ``constraint`` is not None, a Constraint, or a list/tuple of
        Constraints.
    ValueError
        If ``constraint`` is an empty list or tuple.
    """
    if constraint is None or isinstance(constraint, Constraint):
        return constraint

    if not isinstance(constraint, (list, tuple)):
        raise TypeError(
            f"Error: constraint must be a Constraint, a list of Constraints or None; got {type(constraint).__name__}."
        )
    if len(constraint) == 0:
        raise ValueError("Error: constraint is an empty list; pass constraint=None for unconstrained sampling.")

    wrong_types = [type(item).__name__ for item in constraint if not isinstance(item, Constraint)]
    if wrong_types:
        raise TypeError(f"Error: every item in a list of constraints must be a Constraint; got {', '.join(wrong_types)}.")

    return MultiConstraint(list(constraint))


def _relaxation_mode(relax):
    """
    Check generate()'s relax argument and return the relaxation mode it asks for.

    Parameters
    ----------
    relax : str or bool
        "standard", "fast", True (shorthand for "standard", so code written
        before "fast" existed keeps working) or False.

    Returns
    -------
    str or bool
        "standard", "fast", or False for no relaxation.

    Raises
    ------
    ValueError
        If relax is anything else.
    """
    if relax is True:
        return generation.RELAX_STANDARD
    if relax is False:
        return False
    if isinstance(relax, str) and relax in generation.RELAX_MODES:
        return relax
    raise ValueError(
        f"Error: relax must be 'standard', 'fast', True (the same as 'standard') or False; got {relax!r}."
    )


def generate(
    user_input,
    conformations=configs.DEFAULT_NUMBER_CONFS,
    ionic_strength=configs.DEFAULT_IONIC_STRENGTH,
    device=None,
    steps=configs.DEFAULT_STEPS,
    sampler=configs.DEFAULT_SAMPLER,
    return_structures=False,
    batch_size=configs.DEFAULT_BATCH_SIZE,
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
    remove_errors=False,
    relax="standard",
    compile_forces=False,
    seed=None,
):
    """
    Generate STARLING ensembles and distance maps for one or more sequences.

    This is the primary high-level interface for STARLING ensemble generation. It
    normalizes the provided sequences, runs the diffusion sampler, optionally
    performs MDS refinement, and returns ensemble objects or writes them to disk.

    Parameters
    ----------
    user_input : str or os.PathLike or Sequence[str] or Mapping[str, str]
        Input sequences to process. Supported forms include:

        * Path (str or ``os.PathLike``) to a FASTA, TSV, or ``seq.in`` file
          containing name/sequence rows (blank lines in TSV/``seq.in`` files
          are skipped).
        * Raw amino-acid sequence string.
        * List or tuple of sequence strings.
        * Mapping of sequence names to amino-acid sequences.

        Empty sequences and non-canonical residues trigger a
        :class:`ValueError`, except that non-canonical residues in FASTA
        files are converted (X->G, B->N, U->C, Z->Q; ``*``, ``-`` and spaces
        removed) with a warning naming each sequence changed and how many
        residues were converted or removed. Sequences longer than
        ``configs.MAX_SEQUENCE_LENGTH`` (380) residues are skipped with a
        printed warning; if no sequences are left, a :class:`ValueError` is
        raised before any models are loaded.
    conformations : int, default=configs.DEFAULT_NUMBER_CONFS
        Number of conformations to sample per sequence.
    ionic_strength : float, default=configs.DEFAULT_IONIC_STRENGTH
        Ionic strength (mM) supplied to the generative model. Must be a
        finite number >= 0.
    device : str or None, default=None
        Device identifier (``'cuda'``, ``'mps'``, or ``'cpu'``). ``None`` selects the
        best available accelerator.
    steps : int, default=configs.DEFAULT_STEPS
        Number of denoising diffusion steps. DDPM ignores it and always uses
        every training timestep; PLMS needs at least 20 steps (a ValueError is
        raised before the models load otherwise); DDIM warns below 10 steps,
        where its large deterministic steps narrow the ensemble.
    sampler : str, default=configs.DEFAULT_SAMPLER
        Sampling method: ``'ddim'``, ``'ddpm'``, ``'dpmpp'`` (DPM-Solver++(2M)),
        or ``'plms'``. DPM++ is deterministic after initial noise sampling and
        does not support constraints. DDPM always uses the full training schedule.
    return_structures : bool, default=False
        When ``True`` include 3D coordinate ensembles in the results. These
        are relaxed with Mpipi-GG by default (see ``relax``).
    batch_size : int, default=configs.DEFAULT_BATCH_SIZE
        Batch size used for sampling iterations.
    output_directory : str or os.PathLike or None, default=None
        Directory where generated outputs are written. When ``None`` nothing is saved.
        Each ensemble is saved as ``<name>.starling`` (with ``<name>`` the full
        sequence name, including any dots) and, when ``return_structures`` is
        ``True``, its structures as ``<name>_STARLING.pdb`` and
        ``<name>_STARLING.xtc`` (see ``pdb_trajectory``). Because sequence
        names become file names, they may not contain a path separator
        (such as ``/``) when an output directory is given; this is checked
        before any models are loaded.
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
        When ``True`` the structures are written as a single multi-model PDB
        file (``<name>_STARLING.pdb``) instead of a single-frame PDB topology
        plus an XTC trajectory (no XTC file is written). Only applies when
        ``return_structures`` is ``True`` and an ``output_directory`` is
        provided.
    return_single_ensemble : bool, default=False
        When ``True`` and exactly one sequence is processed, return a single
        :class:`starling.structure.ensemble.Ensemble`. Raises :class:`ValueError`
        if multiple sequences are supplied.
    constraint : Constraint or list of Constraint or None, default=None
        Constraint (from :mod:`starling.inference.constraints`) applied during
        sampling. A list or tuple of constraints is combined into a
        :class:`~starling.inference.constraints.MultiConstraint`: each
        constraint keeps its own ``constraint_weight`` and guidance window,
        while the time schedule is MultiConstraint's default (``"cosine"``)
        and the constraints' own ``schedule`` settings are ignored; build a
        MultiConstraint yourself to choose its schedule. Anything else raises
        :class:`TypeError` before any models are loaded. Not supported by the
        ``'dpmpp'`` sampler.
    remove_errors : bool, default=False
        When ``True``, conformers with physically impossible inter-residue
        distances are discarded and replaced with freshly generated ones, so
        that exactly ``conformations`` clean conformers are returned. Screening
        uses :meth:`~starling.structure.ensemble.Ensemble.check_for_errors` on
        the distance maps and, when ``return_structures`` is also True,
        :meth:`~starling.structure.ensemble.Ensemble.check_for_errors_trajectory`
        on the reconstructed 3D conformers. Because filtering happens before
        anything is written to disk, the saved ``.xtc``/``.pdb`` and
        ``.starling`` outputs contain only clean conformers.
    relax : str or bool, default="standard"
        How the 3D conformers are finished when ``return_structures`` is
        ``True``. With ``"standard"`` or ``"fast"``, the conformers
        reconstructed by MDS are relaxed with the Mpipi-GG force field while
        restrained to their STARLING distance maps (see
        :mod:`starling.minimizer`). MDS gets the global shape right but
        compresses bonds and leaves beads too close; relaxation minimizes
        each structure and then thermalizes it briefly at 300 K, giving
        bond, angle and dihedral distributions close to those of Mpipi-GG
        simulations while changing the radius of gyration by ~0.3% on
        average. The two modes relax in exactly the same way and differ only
        in the MDS step before it. ``"standard"`` reconstructs each structure
        as STARLING always has. ``"fast"`` gets the MDS starting point from
        only the eigenvectors it needs and runs 100 rather than 300 SMACOF
        iterations, leaving relaxation to finish the fit; this makes MDS
        about 4x faster for long sequences. The relaxed ensembles match the
        standard ones in fit to the distance maps, radius of gyration, bond
        lengths and closest contacts, but individual conformations differ, because
        each lands on a different structure that fits its map about equally
        well. ``True`` is the same as ``"standard"``; ``False`` returns the
        raw MDS structures, always made with the standard settings.
        Relaxation runs before the trajectory-level error screen when
        ``remove_errors`` is ``True``, so MDS-broken conformers are repaired
        rather than discarded. Has no effect when ``return_structures`` is
        ``False``. Relaxation is on by default (as it is for
        ``starling -r`` on the command line).
    compile_forces : bool, default=False
        Fuse CUDA thermalization forces for sustained throughput. Requires
        relaxation (``relax`` not ``False``) and ``return_structures=True``.
        Adds compilation startup time; preserves precision, the force
        field, and thermalization steps.
    seed : int or None, default=None
        Seed for the random numbers this call draws. With a seed, all of the
        sampler's noise comes from its own generator, so the same seed
        reproduces the same ensembles (on the same hardware and settings)
        whatever else the process has drawn, and torch's global random
        state is left alone. With ``None``, sampling draws from torch's
        global generator, so ``torch.manual_seed`` upstream still makes runs
        reproducible. Either way, relaxation draws from separate seeds
        derived from ``seed`` (or from ``torch.initial_seed()``), so changing
        ``relax`` or ``return_structures`` never changes the distance maps. Must be a non-negative integer below 2**64.
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
        If sequences are empty or contain non-canonical residues (outside FASTA
        files), no sequence is short enough to generate, sequence names
        contain a path separator while an ``output_directory`` is given, or
        arguments or their combinations are invalid.
    TypeError
        If ``constraint`` is not a Constraint, a list of Constraints, or None.

    """
    # check user input, return a sequence dict.
    _sequence_dict = handle_input(user_input, output_name=output_name)

    # we do this specific sanity check EARLY so we don't silently fix what would
    # otherwise be a faulty input
    if return_single_ensemble and len(_sequence_dict) > 1:
        raise ValueError(f"Error: requested single ensemble yet provided input of {len(_sequence_dict)} sequences.")

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

    # fail now, rather than after loading the models, if nothing is left to generate
    if len(sequence_dict) == 0:
        if removed_counter > 0:
            raise ValueError(
                f"Error: all {removed_counter} input sequence(s) are longer than the "
                f"{configs.MAX_SEQUENCE_LENGTH}-residue maximum, so there is nothing to generate."
            )
        raise ValueError("Error: no sequences were provided.")

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
            print(f"[STATUS]: Generating distance maps for {len(sequence_dict)} sequences{bonus_message}.")

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

    # check ionic strength (mM) is a real, finite, non-negative number; bool is
    # excluded because it is a subclass of int
    if isinstance(ionic_strength, bool) or not isinstance(ionic_strength, numbers.Real):
        raise ValueError(f"Error: ionic_strength must be a number (in mM); got {ionic_strength!r}.")
    if not np.isfinite(ionic_strength) or ionic_strength < 0:
        raise ValueError(f"Error: ionic_strength must be finite and >= 0 (in mM); got {ionic_strength!r}.")

    # make sure batch_size is not smaller than conformations.
    # if it is, make batch_size = conformations.
    if batch_size > conformations:
        batch_size = conformations

    # check output_directory is a directory that exists.
    if output_directory is not None:
        if not os.path.exists(output_directory):
            raise FileNotFoundError(f"Error: Directory {output_directory} does not exist.")

        # sequence names become file names, so one containing a path separator
        # (common in FASTA headers) would crash when saving, after sampling
        path_separators = [separator for separator in (os.sep, os.altsep) if separator is not None]
        bad_names = [str(name) for name in sequence_dict if any(sep in str(name) for sep in path_separators)]
        if bad_names:
            raise ValueError(
                f"Error: sequence names are used as output file names, so they cannot contain a path separator "
                f"({' or '.join(path_separators)}) when output_directory is set. Please rename: {', '.join(bad_names)}"
            )

    # check sampler is a string
    if not isinstance(sampler, str):
        raise ValueError("Error: sampler must be a string.")
    if sampler.lower() not in ("ddim", "ddpm", "dpmpp", "plms"):
        raise ValueError("Error: sampler must be one of 'ddim', 'ddpm', 'dpmpp', or 'plms'.")
    if sampler.lower() == "dpmpp" and constraint is not None:
        raise ValueError("DPM++ does not support constraints; use DDIM, PLMS or DDPM")
    # checked here as well as in PLMSSampler so it fails before the models load
    if sampler.lower() == "plms" and steps < PLMS_MIN_STEPS:
        raise ValueError(
            f"PLMS needs at least {PLMS_MIN_STEPS} steps (got {steps}); use DDIM or DPM++ for fewer steps"
        )

    # a list of constraints is combined into one; anything that is not a
    # constraint fails here, before the models are loaded
    constraint = _prepare_constraint(constraint)

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

    # check remove_errors is a bool
    if not isinstance(remove_errors, bool):
        raise ValueError("Error: remove_errors must be True or False.")

    # check relax names a relaxation mode (or is False), and turn True into "standard"
    relax = _relaxation_mode(relax)
    if not isinstance(compile_forces, bool):
        raise ValueError("Error: compile_forces must be True or False.")
    if compile_forces and (not relax or not return_structures):
        raise ValueError("compile_forces requires relaxation (relax='standard' or 'fast') and return_structures=True")

    # check seed is None or an integer torch.Generator.manual_seed accepts
    # (bool is an int subclass, so it is excluded explicitly)
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or not 0 <= seed < MAX_SEED_EXCLUSIVE:
            raise ValueError(f"Error: seed must be None or an integer in [0, 2**64); got {seed!r}.")
        seed = int(seed)

    # we do this specific sanity check to make the logic later in this function easier
    if return_single_ensemble and return_data is False:
        raise ValueError("Error: requested single ensemble yet also did not request data to be returned.")

    if return_data is False and output_directory is None:
        raise ValueError(
            "Error: both no return data (return_data=False) and also did not specifiy an output_directory; this means no output will be returned/saved anywhere, which is probably not desired!"
        )

    # check device, get back a torch.device (not a str!)
    device = utilities.check_device(device)
    if compile_forces and device.type != "cuda":
        raise ValueError("compile_forces requires a CUDA device")

    # run the actual inference and return the results
    ensemble_return = generation.generate_backend(
        sequence_dict,
        conformations,
        device,
        steps,
        sampler,
        return_structures,
        batch_size,
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
        remove_errors=remove_errors,
        relax=relax,
        compile_forces=compile_forces,
        seed=seed,
    )

    if not return_data:
        return None

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
    encoder_path=None,
    ddpm_path=None,
):
    """Encode an ensemble of distance maps into STARLING's latent space.

    Parameters
    ----------
    ensemble : np.ndarray
        Distance maps in Angstroms, shape ``(n_conformers, L, L)`` with
        ``L <= 384`` (for example ``Ensemble.distance_maps()``). Maps smaller
        than 384 x 384 are zero-padded to that size and all maps are
        converted to float32 before encoding.
    batch_size : int, optional
        Number of distance maps encoded at once. Default 32.
    device : str | torch.device | None, optional
        Device hint forwarded to :func:`utilities.check_device`. ``None`` lets
        STARLING pick the best available accelerator.
    encoder_path : str | None, optional
        Override the default encoder checkpoint.
    ddpm_path : str | None, optional
        Override the default diffusion checkpoint used by the shared model
        manager.

    Returns
    -------
    np.ndarray
        Latents (the mode of the VAE's posterior) of shape
        ``(n_conformers, 24, 24)``. Nothing is written to disk; save the
        array yourself (e.g. with ``np.save``) if needed.
    """
    # check device, get back a torch.device (not a str!)
    device = utilities.check_device(device)

    embeddings = generation.ensemble_encoder_backend(
        ensemble=ensemble,
        device=device,
        batch_size=batch_size,
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
        Directory for optional on-disk exports; each embedding is saved as
        ``<name>.pt`` and nothing is returned. Leave ``None`` to keep
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
        Sequences are always sorted by length (longest first) before
        batching, which already keeps padding low. When ``True``, batches
        are also split wherever the length bucket (``L // bucket_size``)
        changes, so no batch mixes buckets and no sequence is padded by
        ``bucket_size`` residues or more, at the cost of more (smaller)
        batches.
    bucket_size : int, optional
        Width of each length bucket in residues when ``bucket`` is
        ``True``. Ignored otherwise.
    free_cuda_cache : bool, optional
        Release CUDA memory after each batch for long inference jobs.
    return_on_cpu : bool, optional
        Convert embeddings to CPU tensors before returning. Set to ``False`` to
        keep them on the selected device for downstream GPU workflows.

    Returns
    -------
    dict[str, torch.Tensor] or None
        Mapping from sequence identifiers to embedding tensors. The trailing
        tensor shape is ``(L, D)`` for residue-level embeddings or ``(D,)`` for
        aggregated embeddings (``aggregate=True``), where ``L`` is sequence
        length and ``D`` is the latent dimension. ``None`` when
        ``output_directory`` is set, as the embeddings are written to disk
        instead.

    Notes
    -----
    The encoder shares weights with the ensemble generator, so successive calls
    reuse cached models through ``generation.model_manager``. Use
    ``encoder_path`` and ``ddpm_path`` to experiment with fine-tuned weights
    without mutating global configuration.
    """
    # check device, get back a torch.device (not a str!)
    device = utilities.check_device(device)

    if not pretokenized:
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
