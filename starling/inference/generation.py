import gc
import os
import time
from datetime import datetime

import numpy as np
import torch
from soursop.sstrajectory import SSTrajectory
from tqdm.auto import tqdm

from starling import configs
from starling.data.tokenizer import StarlingTokenizer
from starling.inference.model_loading import ModelManager
from starling.minimizer import relax_conformations
from starling.samplers.ddim_sampler import DDIMSampler
from starling.samplers.ddpm_sampler import DDPMSampler
from starling.samplers.plms_sampler import PLMSSampler
from starling.structure.coordinates import (
    create_ca_topology_from_coords,
    generate_3d_coordinates_from_distances,
)
from starling.structure.ensemble import Ensemble

# initialize model_manager singleton. This happens when this module
# is imported to ensemble_generation, so we can use the
# same model_manager for all calls to generate_backend.
model_manager = ModelManager()


def symmetrize_distance_map(dist_map):
    """
    Symmetrize a distance map by replacing the lower triangle with the upper triangle values.

    Parameters
    ----------
    dist_map : torch.Tensor
        A 2D tensor representing the distance map.

    Returns
    -------
    torch.Tensor
        A symmetrized distance map.

    """

    # Ensure the distance map is 2D
    dist_map = dist_map.squeeze(0) if dist_map.dim() == 3 else dist_map

    # Create a copy of the distance map to modify
    sym_dist_map = dist_map.clone()

    # Replace the lower triangle with the upper triangle values
    mask_upper_triangle = torch.triu(torch.ones_like(dist_map), diagonal=1).bool()
    mask_lower_triangle = ~mask_upper_triangle

    # Set lower triangle values to be the same as the upper triangle
    sym_dist_map[mask_lower_triangle] = dist_map.T[mask_lower_triangle]

    # Set diagonal values to zero
    sym_dist_map.fill_diagonal_(0)

    return sym_dist_map.cpu()


def sequence_encoder_backend(
    sequence_dict,
    device,
    batch_size,
    ionic_strength,
    aggregate=True,
    output_directory=None,
    model_manager=model_manager,
    encoder_path=None,
    ddpm_path=None,
    pretokenized: bool = False,
    bucket: bool = False,
    bucket_size: int = 32,
    free_cuda_cache: bool = False,
    return_on_cpu: bool = True,
):
    """
    Generate embeddings for sequences and optionally save them to disk.

    Parameters
    ----------
    sequence_dict : dict
        Dictionary of sequence names to sequences
    device : str
        Device to use for computation
    batch_size : int
        Batch size for processing
    ionic_strength : float
        Ionic strength [mM] to condition the model
    output_directory : str, optional
        If provided, embeddings will be saved to this directory with sequence name as filename
    model_manager : ModelManager
        Model manager instance
    encoder_path : str, optional
        Custom encoder path
    ddpm_path : str, optional
        Custom diffusion model path
    pretokenized : bool, default False
        If True, values of sequence_dict are assumed to already be iterable collections
        of integer token ids (lists/tuples/torch tensors). Skips tokenization.
    bucket : bool, default False
        If True, sequences are grouped into coarse length buckets (multiple of bucket_size)
        to reduce padding waste. Beneficial when length distribution is very broad.
    bucket_size : int, default 32
        Length resolution for bucketing when bucket=True. Sequences with lengths that
        fall into the same bucket ( (L//bucket_size) ) are batched together.
    free_cuda_cache : bool, default False
        If True and running on CUDA, calls torch.cuda.empty_cache() after each batch.
    return_on_cpu : bool, default True
        If True, embeddings are transferred to CPU before being returned or saved.
        If False, embeddings remain on the original device (e.g., GPU), which can be
        useful when performing downstream tensor operations on the same device.

    Returns
    -------
    dict or None
        If output_directory is None, returns dictionary name -> tensor (L_i, D).
        Otherwise returns None (embeddings written to disk as <name>.pt).
    """
    tokenizer = None if pretokenized else StarlingTokenizer()
    _, diffusion = model_manager.get_models(
        device=device, encoder_path=encoder_path, ddpm_path=ddpm_path
    )

    ionic_strength = torch.tensor([ionic_strength], device=device).unsqueeze(0)

    # Prepare output handling
    if output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)
        print(f"Saving embeddings to: {os.path.abspath(output_directory)}")
        embedding_dict = None
    else:
        embedding_dict = {}

    # Normalize sequences -> (name, token_list)
    prepared = []
    for name, seq in sequence_dict.items():
        if pretokenized:
            # Accept list/tuple/torch.Tensor of ints
            if isinstance(seq, torch.Tensor):
                tokens = seq.tolist()
            else:
                tokens = list(seq)
        else:
            tokens = tokenizer.encode(seq)
        prepared.append((name, tokens))

    # Optional bucketing to reduce padding
    if bucket:
        bucketed = {}
        for name, toks in prepared:
            key = (len(toks) // bucket_size) * bucket_size
            bucketed.setdefault(key, []).append((name, toks))
        # Flatten buckets ordered by descending key (longer first)
        ordered = []
        for key in sorted(bucketed.keys(), reverse=True):
            # within bucket sort descending length
            ordered.extend(sorted(bucketed[key], key=lambda x: len(x[1]), reverse=True))
        prepared = ordered
    else:
        # Just sort by length descending
        prepared.sort(key=lambda x: len(x[1]), reverse=True)

    names = [n for n, _ in prepared]
    seqs = [t for _, t in prepared]
    total = len(seqs)
    lengths = [len(t) for t in seqs]

    _inference_ctx = (
        torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    )

    with _inference_ctx():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_sequences = seqs[start:end]
            batch_names = names[start:end]
            batch_lengths = lengths[start:end]

            current_bs = end - start
            max_length = batch_lengths[0]

            # Build tensors (only CPU→GPU transfer)
            sequence_tensor = torch.zeros(
                (current_bs, max_length), dtype=torch.long, device=device
            )
            attention_mask = torch.zeros(
                (current_bs, max_length), dtype=torch.bool, device=device
            )

            for i, (seq_tokens, length_i) in enumerate(
                zip(batch_sequences, batch_lengths)
            ):
                sequence_tensor[i, :length_i] = torch.as_tensor(
                    seq_tokens, dtype=torch.long, device=device
                )
                attention_mask[i, :length_i] = True

            batch_embeddings = diffusion.sequence2labels(
                sequences=sequence_tensor,
                sequence_mask=attention_mask,
                ionic_strength=ionic_strength,
            )

            # Transfer to CPU if requested
            if return_on_cpu:
                batch_embeddings = batch_embeddings.cpu()

            # Process results
            for i, (name, length_i) in enumerate(zip(batch_names, batch_lengths)):
                emb = batch_embeddings[i, :length_i]

                if aggregate:
                    emb = emb.mean(axis=0)

                if output_directory is not None:
                    torch.save(emb, os.path.join(output_directory, f"{name}.pt"))
                else:
                    embedding_dict[name] = emb

            del batch_embeddings, sequence_tensor, attention_mask
            if (
                free_cuda_cache
                and torch.cuda.is_available()
                and device.startswith("cuda")
            ):
                torch.cuda.empty_cache()

    return embedding_dict


def ensemble_encoder_backend(
    ensemble,
    device,
    batch_size,
    output_directory=None,
    model_manager=model_manager,
    encoder_path=None,
    ddpm_path=None,
):
    encoder_model, diffusion = model_manager.get_models(
        device=device, encoder_path=encoder_path, ddpm_path=ddpm_path
    )

    assert isinstance(ensemble, np.ndarray), (
        "ensemble must be a numpy array. If you have a torch tensor, convert it to numpy first."
    )
    assert ensemble.ndim == 3, "ensemble must be a 3D array (batch, height, width)."

    if ensemble.shape[2] != 384:
        H_pad = max(0, 384 - ensemble.shape[1])  # vertical padding (bottom)
        W_pad = max(0, 384 - ensemble.shape[2])  # horizontal padding (right)

        ensemble = np.pad(
            ensemble,
            pad_width=(
                (0, 0),
                (0, H_pad),
                (0, W_pad),
            ),  # (N axis, bottom of H axis, right of W axis)
            mode="constant",
            constant_values=0,
        )

    # get num_batches and remaining samples
    num_batches = ensemble.shape[0] // batch_size
    remaining_samples = ensemble.shape[0] % batch_size

    latent_spaces = []

    # real_batch_count no longer needed (legacy from previous implementation)

    ensemble = torch.from_numpy(ensemble)
    ensemble = ensemble.unsqueeze(1)  # Add a channel dimension

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = (batch + 1) * batch_size

        batch_ensemble = ensemble[start_idx:end_idx]

        latent_space = encoder_model.encode(
            batch_ensemble.to(device),
        ).mode()

        latent_spaces.append(latent_space.detach().squeeze().cpu().numpy())

    if remaining_samples > 0:
        start_idx = num_batches * batch_size
        batch_ensemble = ensemble[start_idx:]

        latent_space = encoder_model.encode(
            batch_ensemble.to(device),
        ).mode()

        latent_spaces.append(latent_space.detach().squeeze().cpu().numpy())

    return np.concatenate(latent_spaces)


# Maximum number of sample-and-filter rounds attempted when remove_errors is
# enabled. This is a backstop so a pathological sequence (where the model keeps
# producing unphysical conformers) cannot spin forever.
DEFAULT_MAX_ERROR_FILTER_ROUNDS = 10


def _sample_distance_maps(
    sampler,
    sequence,
    conformations,
    batch_size,
    show_per_step_progress_bar,
    constraint,
):
    """
    Sample a set of symmetrized distance maps for a single sequence.

    This is the batching loop used by :func:`generate_backend`, factored out so
    it can be called repeatedly when erroneous conformers are being filtered and
    replaced.

    Parameters
    ----------
    sampler : object
        An initialized sampler (``DDIMSampler``, ``PLMSSampler`` or
        ``DDPMSampler``) exposing a ``.sample()`` method.

    sequence : str
        The amino acid sequence being sampled.

    conformations : int
        Number of distance maps to sample.

    batch_size : int
        Number of conformations sampled per batch.

    show_per_step_progress_bar : bool
        Whether the sampler should show its per-step progress bar.

    constraint : object or None
        Optional constraint object passed through to the sampler.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(conformations, L, L)`` holding the symmetrized
        distance maps, where ``L`` is the sequence length.
    """
    num_batches = conformations // batch_size
    remaining_samples = conformations % batch_size

    if remaining_samples > 0:
        real_batch_count = num_batches + 1
    else:
        real_batch_count = num_batches

    starling_dm = []

    for batch in range(num_batches):
        distance_maps = sampler.sample(
            batch_size,
            labels=sequence,
            show_per_step_progress_bar=show_per_step_progress_bar,
            batch_count=batch + 1,
            max_batch_count=real_batch_count,
            constraint=constraint,
        )
        starling_dm.append(
            [
                symmetrize_distance_map(dm[:, : len(sequence), : len(sequence)])
                for dm in distance_maps
            ]
        )

    if remaining_samples > 0:
        distance_maps = sampler.sample(
            remaining_samples,
            labels=sequence,
            show_per_step_progress_bar=show_per_step_progress_bar,
            batch_count=real_batch_count,
            max_batch_count=real_batch_count,
            constraint=constraint,
        )
        starling_dm.append(
            [
                symmetrize_distance_map(dm[:, : len(sequence), : len(sequence)])
                for dm in distance_maps
            ]
        )

    return torch.cat([torch.stack(batch) for batch in starling_dm], dim=0)


def _relax_coordinates(
    sequence,
    coordinates,
    distance_maps,
    ionic_strength,
    device,
    batch_size,
    show_progress_bar,
    verbose=False,
):
    """
    Relax MDS-reconstructed conformers with Mpipi-GG.

    Thin wrapper around :func:`starling.minimizer.relax_conformations` that
    works in the nanometre coordinates used throughout this module. Each
    conformer is restrained to the STARLING distance map it was built from, so
    local geometry (bond lengths, overlapping beads) is repaired while the
    global shape is kept.

    Parameters
    ----------
    sequence : str
        The amino acid sequence.

    coordinates : np.ndarray
        MDS coordinates in nanometres, shape ``(n_conformers, L, 3)``.

    distance_maps : np.ndarray
        The STARLING distance maps (Angstroms) the coordinates were built
        from, shape ``(n_conformers, L, L)``.

    ionic_strength : float
        Ionic strength in mM, which sets the Debye length.

    device : torch.device or str
        Device to run the minimization on.

    batch_size : int
        Number of conformers minimized together.

    show_progress_bar : bool
        Whether to show the relaxation progress bar.

    verbose : bool
        If True, print a before/after summary of the relaxation.

    Returns
    -------
    np.ndarray
        Relaxed coordinates in nanometres, shape ``(n_conformers, L, 3)``.
    """
    result = relax_conformations(
        np.asarray(coordinates) * configs.CONVERT_ANGSTROM_TO_NM,
        sequence,
        reference_distances=distance_maps,
        ionic_strength=ionic_strength,
        batch_size=batch_size,
        device=device,
        progress_bar=show_progress_bar,
    )

    if verbose:
        print(result.summary())

    return result.coordinates / configs.CONVERT_ANGSTROM_TO_NM


def _generate_error_filtered_conformers(
    sampler,
    sequence,
    conformations,
    batch_size,
    show_per_step_progress_bar,
    constraint,
    return_structures,
    device,
    num_cpus_mds,
    num_mds_init,
    show_progress_bar,
    verbose=False,
    max_rounds=DEFAULT_MAX_ERROR_FILTER_ROUNDS,
    relax=False,
    ionic_strength=configs.DEFAULT_IONIC_STRENGTH,
):
    """
    Generate exactly ``conformations`` conformers that pass the error checks.

    Conformers are sampled, screened, and any that fail are discarded and
    replaced by fresh samples, repeating until the requested number of clean
    conformers has been accumulated (or ``max_rounds`` rounds have elapsed).

    Screening happens in two stages, mirroring the two public checks on
    :class:`~starling.structure.ensemble.Ensemble`:

    1. :meth:`Ensemble.check_for_errors` screens the raw STARLING distance maps
       for physically impossible inter-residue distances.
    2. :meth:`Ensemble.check_for_errors_trajectory` screens the reconstructed 3D
       conformers, catching artefacts introduced by the SMACOF embedding itself.

    Running the cheap distance-map check first means we only pay for 3D
    reconstruction on conformers that already look physically plausible. The
    second stage runs only when ``return_structures`` is True, since without a
    reconstructed trajectory there is nothing for it to inspect.

    If ``relax`` is True, the reconstructed conformers are relaxed with
    Mpipi-GG *before* the trajectory-level screen. Relaxation repairs most
    MDS artefacts (including broken bonds), so those conformers are kept
    rather than discarded, and the screen checks exactly the structures that
    will be returned.

    Parameters
    ----------
    sampler : object
        An initialized sampler exposing a ``.sample()`` method.

    sequence : str
        The amino acid sequence being sampled.

    conformations : int
        Number of *clean* conformations to return.

    batch_size : int
        Number of conformations sampled per batch.

    show_per_step_progress_bar : bool
        Whether the sampler should show its per-step progress bar.

    constraint : object or None
        Optional constraint object passed through to the sampler.

    return_structures : bool
        If True, 3D conformers are reconstructed and the trajectory-level check
        is applied in addition to the distance-map check.

    device : torch.device
        Device used for the MDS reconstruction.

    num_cpus_mds : int
        Number of CPUs available for MDS reconstruction.

    num_mds_init : int
        Number of MDS initializations run in parallel.

    show_progress_bar : bool
        Whether to show the reconstruction progress bar.

    verbose : bool
        If True, report how many conformers were discarded in each round.

    max_rounds : int
        Maximum number of sample-and-filter rounds before giving up. Guards
        against an unbounded loop if the model cannot produce clean conformers
        for a given sequence.

    relax : bool
        If True (and ``return_structures`` is True), relax the reconstructed
        conformers with Mpipi-GG before screening them. Default False.

    ionic_strength : float
        Ionic strength in mM used for the relaxation. Default is STARLING's
        default ionic strength.

    Returns
    -------
    tuple
        ``(distance_maps, coordinates, n_discarded)`` where ``distance_maps`` is
        a numpy array of shape ``(conformations, L, L)``, ``coordinates`` is a
        numpy array of shape ``(conformations, L, 3)`` in nanometres (or None if
        ``return_structures`` is False), and ``n_discarded`` is the total number
        of conformers thrown away across all rounds.

    Raises
    ------
    RuntimeError
        If ``max_rounds`` rounds complete without accumulating the requested
        number of clean conformers.
    """
    kept_maps = []
    kept_coords = []
    n_kept = 0
    n_discarded = 0

    for round_index in range(max_rounds):
        n_needed = conformations - n_kept
        if n_needed <= 0:
            break

        if verbose and round_index > 0:
            print(
                f"  [error filter] round {round_index + 1}: regenerating "
                f"{n_needed} conformation(s) to replace discarded ones"
            )

        sampled = _sample_distance_maps(
            sampler,
            sequence,
            n_needed,
            batch_size,
            show_per_step_progress_bar,
            constraint,
        )

        # --- stage 1: screen the raw distance maps (cheap, pre-reconstruction)
        candidate_maps = sampled.detach().cpu().numpy()
        n_candidates = len(candidate_maps)

        staging = Ensemble(candidate_maps, sequence)
        staging.check_for_errors(remove_errors=True, verbose=False)
        candidate_maps = staging.distance_maps()

        if len(candidate_maps) == 0:
            n_discarded += n_candidates
            continue

        # --- stage 2: screen the reconstructed 3D conformers
        if return_structures:
            coordinates = generate_3d_coordinates_from_distances(
                device,
                batch_size,
                num_cpus_mds,
                num_mds_init,
                candidate_maps,
                progress_bar=show_progress_bar,
            )

            if relax:
                coordinates = _relax_coordinates(
                    sequence,
                    coordinates,
                    candidate_maps,
                    ionic_strength,
                    device,
                    batch_size,
                    show_progress_bar,
                    verbose=verbose,
                )

            ssprotein = SSTrajectory(
                TRJ=create_ca_topology_from_coords(sequence, coordinates)
            ).proteinTrajectoryList[0]

            staging = Ensemble(candidate_maps, sequence, ssprot_ensemble=ssprotein)
            staging.check_for_errors_trajectory(remove_errors=True, verbose=False)

            candidate_maps = staging.distance_maps()
            if len(candidate_maps) == 0:
                n_discarded += n_candidates
                continue

            # xyz is in nanometres, which is exactly what
            # create_ca_topology_from_coords() expects back, so this round-trips
            kept_coords.append(staging.trajectory.traj.xyz)

        kept_maps.append(candidate_maps)
        n_kept += len(candidate_maps)
        n_discarded += n_candidates - len(candidate_maps)

    if n_kept < conformations:
        raise RuntimeError(
            f"Unable to generate {conformations} error-free conformation(s) for "
            f"sequence of length {len(sequence)} after {max_rounds} rounds "
            f"(obtained {n_kept}, discarded {n_discarded}). This usually means "
            f"the model is producing unphysical conformers for this sequence. "
            f"Re-run without error filtering to inspect the raw output, or "
            f"increase the number of allowed rounds."
        )

    # a round may overshoot, so trim back to exactly what was asked for
    final_maps = np.concatenate(kept_maps, axis=0)[:conformations]

    if return_structures:
        final_coords = np.concatenate(kept_coords, axis=0)[:conformations]
    else:
        final_coords = None

    return final_maps, final_coords, n_discarded


def generate_backend(
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
    model_manager=model_manager,
    ionic_strength=150,
    constraint=None,
    encoder_path=None,
    ddpm_path=None,
    remove_errors=False,
    max_error_filter_rounds=DEFAULT_MAX_ERROR_FILTER_ROUNDS,
    relax=False,
):
    """
    Backend function for generating the distance maps using STARLING.

    NOTE - this function does VERY littel sanity checking; to actually perform
    predictions use starling.frontend.ensemble_generation. This is NOT
    a user facing function!

    Parameters
    ---------------
    sequence_dict : dict
        A dictionary with the sequence names as the key and the
       sequences as the values. These names will be used to write
       any output files (if writing is requested).

    ddpm : str
        The path to the DDPM model

    device : str
        The device to use for predictions.

    steps : int
        The number of steps to run the DDPM model.

    ddim : bool
        Whether to use DDIM for sampling.

    return_structures : bool
        Whether to return the 3D structure.

    batch_size : int
        The batch size to use for sampling.

    num_cpus_mds : int
        The number of CPUs to use for MDS. There
        is no point specifying more than the default
        number of MDS runs performed (defined in configs)

    output_directory : str or None
        If None, no output is saved.
        If not None, will save the output to the specified path.
        This includes the distance maps and if return_structures=True,
        the 3D structures.
        The distance maps are saved as .npy files with the names
        <sequence_name>_STARLING_DM.npy
        and the structures are save with the file names
        <sequence_name>_STARLING.xtc and <sequence_name>_STARLING.pdb.

    return_data : bool
        If True, will return the distance maps and structures (if generated)
        as a dictionary regardless of the output_directory. If False, will
        return None. Note the reason to set this to None is if you're
        predicting a large set of sequences this will save memory.

    verbose : bool
        Whether to print verbose output. Default is False.

    show_progress_bar : bool
        Whether to show a progress bar. Default is True.

    show_per_step_progress_bar : bool, optional
        whether to show progress bar per step.
    pdb_trajectory: bool
        Whether to save the trajectory as a PDB file. Default is False.

    model_manager : ModelManager
        A ModelManager object to manage loaded models.
        This lets us avoid loading the model iteratively
        when calling generate multiple times in a single
        session. Default is model_manager, which is initialized
        outside of this function code block. To update the path
        to the models, update the paths in config.py, which are
        read into the ModelManager object located the
        model_loading.py

    encoder_path : str, optional
        Path to a custom encoder model checkpoint file to use instead of the default.
        Default is None, which uses the default model path from configs.py.

    ddpm_path : str, optional
        Path to a custom diffusion model checkpoint file to use instead of the default.
        Default is None, which uses the default model path from configs.py.

    relax : bool, optional
        If True, and return_structures is True, the MDS-reconstructed 3D
        structures are relaxed with Mpipi-GG while restrained to their
        STARLING distance maps (see starling.minimizer). This fixes bond
        lengths and overlapping beads while keeping global dimensions. Has
        no effect if return_structures is False. Default is False here; the
        user-facing generate() defaults to True.

    Returns
    ---------------
    dict or None:
        A dict with the sequence names as the key and
        a starling.ensembl.Ensemble objects for each
        sequence as values.

        If output_directory is not none, the output will save to
        the specified path.
    """

    overall_start_time = time.time()

    # get models. This will only load once even if we call this
    # function multiple times.
    encoder_model, diffusion = model_manager.get_models(
        device=device, encoder_path=encoder_path, ddpm_path=ddpm_path
    )

    # Construct a sampler
    if sampler.lower() == "plms":
        print("Using PLMS sampler")
        sampler = PLMSSampler(
            ddpm_model=diffusion,
            encoder_model=encoder_model,
            n_steps=steps,
            ionic_strength=ionic_strength,
        )
    elif sampler.lower() == "ddim":
        print("Using DDIM sampler")
        sampler = DDIMSampler(
            ddpm_model=diffusion,
            encoder_model=encoder_model,
            n_steps=steps,
            ionic_strength=ionic_strength,
        )
    elif sampler.lower() == "ddpm":
        print("Using DDPM sampler")
        sampler = DDPMSampler(
            ddpm_model=diffusion,
            encoder_model=encoder_model,
            ionic_strength=ionic_strength,
        )
    else:
        raise ValueError(
            f"Error: sampler must be one of 'plms', 'ddim', or 'ddpm'. Got {sampler}."
        )

    # dictionary to hold distance maps and structures if applicable.
    output_dict = {}

    # see if a progress bar is wanted. If it is, set it up.
    # position here is 0, so it will be the first progress bar
    if show_progress_bar:
        pbar = tqdm(
            total=len(sequence_dict),
            position=0,
            desc="Progress through sequences",
            leave=True,
        )

    # iterate over sequence_dict
    for num, seq_name in enumerate(sequence_dict):
        ## -----------------------------------------
        ## Start of prediction cycle for this sequence

        start_time_prediction = time.time()

        # get sequence
        sequence = sequence_dict[seq_name]

        if remove_errors:
            # sample, screen and top-up until we have the requested number of
            # physically plausible conformers
            (
                final_distance_maps,
                filtered_coordinates,
                n_discarded,
            ) = _generate_error_filtered_conformers(
                sampler,
                sequence,
                conformations,
                batch_size,
                show_per_step_progress_bar,
                constraint,
                return_structures,
                device,
                num_cpus_mds,
                num_mds_init,
                show_progress_bar,
                verbose=verbose,
                max_rounds=max_error_filter_rounds,
                relax=relax,
                ionic_strength=ionic_strength,
            )

            end_time_prediction = time.time()
            start_time_structure_generation = time.time()

            if return_structures:
                ssprotein = SSTrajectory(
                    TRJ=create_ca_topology_from_coords(sequence, filtered_coordinates)
                ).proteinTrajectoryList[0]
            else:
                ssprotein = None

            end_time_structure_generation = time.time()

            if verbose:
                print(
                    f"Error filtering: discarded {n_discarded} erroneous "
                    f"conformation(s); returning {len(final_distance_maps)} "
                    f"clean conformation(s)."
                )

        else:
            sym_distance_maps = _sample_distance_maps(
                sampler,
                sequence,
                conformations,
                batch_size,
                show_per_step_progress_bar,
                constraint,
            )

            end_time_prediction = time.time()

            # set time at which we start structure generation to 0
            start_time_structure_generation = time.time()

            # we initialize this to 0 and will update as needed (or not)
            end_time_structure_generation = time.time()

            # do ensemble reconstruction if requested
            if return_structures:
                coordinates = generate_3d_coordinates_from_distances(
                    device,
                    batch_size,
                    num_cpus_mds,
                    num_mds_init,
                    sym_distance_maps,
                    progress_bar=show_progress_bar,
                )

                # repair MDS artefacts in local geometry, holding each structure
                # to the distance map it was built from
                if relax:
                    coordinates = _relax_coordinates(
                        sequence,
                        coordinates,
                        sym_distance_maps.detach().cpu().numpy(),
                        ionic_strength,
                        device,
                        batch_size,
                        show_progress_bar,
                        verbose=verbose,
                    )

                # make traj as an sstrajectory object and extract out the ssprotein object
                ssprotein = SSTrajectory(
                    TRJ=create_ca_topology_from_coords(sequence, coordinates)
                ).proteinTrajectoryList[0]

                end_time_structure_generation = time.time()

            # if no structures are requested, set ssprotein to None
            else:
                ssprotein = None

            # pull the distance maps out of the tensor and convert to numpy
            final_distance_maps = sym_distance_maps.detach().cpu().numpy()

        # capture this now: final_distance_maps is deleted below when
        # return_data is False, but the verbose summary still needs the count
        n_conformers = len(final_distance_maps)

        E = Ensemble(final_distance_maps, sequence, ssprot_ensemble=ssprotein)

        # if we are saving things, save as we progress through so we generate
        # structures/DMs in situ
        if output_directory is not None:
            # num == 0 just means we are on the first sequence.
            if verbose and num == 0:
                print(f"Saving results to: {os.path.abspath(output_directory)}")

            # if we're saving structures do that first;
            if return_structures:
                # this saves both a topology (PDB) and a trajectory (XTC) file
                E.save_trajectory(
                    filename_prefix=os.path.join(
                        output_directory, seq_name + "_STARLING"
                    ),
                    pdb_trajectory=pdb_trajectory,
                )

            # save full ensemble
            E.save(os.path.join(output_directory, f"{seq_name}"))

        ## End of prediction cycle for this sequence
        ## -----------------------------------------

        # if we are returning data, add the data to the output_dict
        if return_data:
            output_dict[seq_name] = E

        # if not, force cleanup of things to save memory
        else:
            del E
            del final_distance_maps
            gc.collect()

        # update progress bar if we have one.
        if show_progress_bar:
            pbar.update(1)

        if verbose:
            elapsed_time_structure_generation = (
                end_time_structure_generation - start_time_structure_generation
            )
            elapsed_time_prediction = end_time_prediction - start_time_prediction
            total_time = elapsed_time_structure_generation + elapsed_time_prediction

            print(
                f"\n\n##### SUMMARY OF SEQUENCE PREDICTION ({num + 1}/{len(sequence_dict)}) #####"
            )
            print(f"Sequence name                       : {seq_name}")
            print(f"Sequence length                     : {len(sequence)}")
            print(f"Number of conformers                : {n_conformers}")
            print(f"Number of steps                     : {steps}")
            print(
                f"Total time for prediction           : {round(elapsed_time_prediction, 2)}s ({round(100 * (elapsed_time_prediction / total_time), 2)}% of time)"
            )
            print(
                f"Total time for structure generation : {round(elapsed_time_structure_generation, 2)}s ({round(100 * (elapsed_time_structure_generation / total_time), 2)}% of time)"
            )
            print(f"Time per conformer                  : {total_time / n_conformers}s")
            print("\n")
        else:
            # changed from print to pass because if not verbose, we don't need to do anything.
            pass

    # make sure we close the progress bar if we used one
    if show_progress_bar:
        pbar.close()

    if verbose:
        # Convert total time to hours, minutes, and seconds
        overall_time = time.time() - overall_start_time
        total_hours = round(overall_time // 3600, 2)
        total_minutes = round((overall_time % 3600) // 60, 2)
        total_seconds = round(overall_time % 60, 2)
        print("-------------------------------------------------------")
        print(f"Summary of all predictions ({len(sequence_dict)} sequences)")
        print("-------------------------------------------------------")
        print(
            f"\nTotal time (all sequences, all I/O) : {total_hours} hrs {total_minutes} mins {total_seconds} secs"
        )

        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        print("STARLING predictions completed at:", formatted_datetime)
        print("")

    return output_dict
