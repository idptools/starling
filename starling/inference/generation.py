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
from starling.inference.model_loading import ModelManager, weights_references
from starling.minimizer import relax_conformations
from starling.samplers import SAMPLERS
from starling.structure.coordinates import (
    DEFAULT_SMACOF_ITERATIONS,
    FULL_EIGENDECOMPOSITION,
    SUBSPACE_ITERATION,
    _mds_coordinates,
    create_ca_topology_from_coords,
    smacof_converged,
)
from starling.structure.weighted_stress import map_error_weights
from starling.structure.ensemble import Ensemble
from starling.utilities import (
    MODEL_FLOAT32_MATMUL_PRECISION,
    float32_matmul_precision,
    symmetrize_tensor_distance_maps,
)

# Relaxation modes generate() accepts. Both relax every structure in exactly
# the same way; they differ only in how much work SMACOF does beforehand (see
# _smacof_settings). "standard" rebuilds each structure as STARLING always
# has; "fast" starts SMACOF from a faster but equivalent classical-MDS
# solution and stops it after FAST_RELAX_SMACOF_ITERATIONS iterations.
RELAX_STANDARD = "standard"
RELAX_FAST = "fast"
RELAX_MODES = (RELAX_STANDARD, RELAX_FAST)

# SMACOF iterations before relax="fast". Relaxation fits each structure to the
# same distance map again, so SMACOF only needs to get the global shape right
# first. SMACOF does most of its work in its first few dozen iterations; later
# ones mostly move structures about an Angstrom without improving the fit.
# Across 20-380 residues and compact to expanded ensembles, structures relaxed
# after 100 iterations fit their maps, and match Rg, bond lengths and
# closest contacts, as well as those relaxed after DEFAULT_SMACOF_ITERATIONS (see
# test_fast_relaxation_matches_standard_relaxation in test_smacof_envelope.py).
FAST_RELAX_SMACOF_ITERATIONS = 100

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

    return symmetrize_tensor_distance_maps(dist_map).cpu()


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
    aggregate : bool, default True
        If True, each sequence's residue-level embeddings (L_i, D) are
        mean-pooled over residues into a single (D,) vector.
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
        Sequences are always sorted by length (longest first) before batching,
        which already keeps padding low. If True, batches are additionally split
        wherever the length bucket (L // bucket_size) changes, so a batch never
        mixes buckets and no sequence in it is padded by bucket_size residues or
        more, at the cost of more (smaller) batches.
    bucket_size : int, default 32
        Width of each length bucket in residues when bucket=True.
    free_cuda_cache : bool, default False
        If True and running on CUDA, calls torch.cuda.empty_cache() after each batch.
    return_on_cpu : bool, default True
        If True, embeddings are transferred to CPU before being returned or saved.
        If False, embeddings remain on the original device (e.g., GPU), which can be
        useful when performing downstream tensor operations on the same device.

    Returns
    -------
    dict or None
        If output_directory is None, returns dictionary name -> tensor, of
        shape (L_i, D) or, when aggregate=True, (D,). Otherwise returns None
        (embeddings written to disk as <name>.pt).
    """
    tokenizer = None if pretokenized else StarlingTokenizer()
    _, diffusion = model_manager.get_models(device=device, encoder_path=encoder_path, ddpm_path=ddpm_path)

    # float32 whatever type it arrives as: a numpy float64 would otherwise
    # make a float64 tensor, which the float32 model rejects (and MPS lacks)
    ionic_strength = torch.tensor([[float(ionic_strength)]], dtype=torch.float32, device=device)

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
            assert tokenizer is not None
            tokens = tokenizer.encode(seq)
        prepared.append((name, tokens))

    # Sort by length descending so each batch holds sequences of similar length
    # (and is padded to the length of its first sequence)
    prepared.sort(key=lambda x: len(x[1]), reverse=True)

    # Optional bucketing: never let a batch span two length buckets. As
    # prepared is sorted, buckets come out longest first.
    if bucket:
        buckets: dict[int, list] = {}
        for name, toks in prepared:
            buckets.setdefault(len(toks) // bucket_size, []).append((name, toks))
        groups = list(buckets.values())
    else:
        groups = [prepared]

    batches = []
    for group in groups:
        for start in range(0, len(group), batch_size):
            batches.append(group[start : start + batch_size])

    _inference_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad

    with _inference_ctx(), float32_matmul_precision(MODEL_FLOAT32_MATMUL_PRECISION):
        for batch in batches:
            batch_names = [name for name, _ in batch]
            batch_sequences = [toks for _, toks in batch]
            batch_lengths = [len(toks) for toks in batch_sequences]

            current_bs = len(batch)
            max_length = batch_lengths[0]

            # Build tensors (only CPU→GPU transfer)
            sequence_tensor = torch.zeros((current_bs, max_length), dtype=torch.long, device=device)
            attention_mask = torch.zeros((current_bs, max_length), dtype=torch.bool, device=device)

            for i, (seq_tokens, length_i) in enumerate(zip(batch_sequences, batch_lengths)):
                sequence_tensor[i, :length_i] = torch.as_tensor(seq_tokens, dtype=torch.long, device=device)
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
                # clone, because the slice is a view of the whole padded batch:
                # torch.save would write all of it, and a stored view keeps the
                # whole batch alive (on the GPU if return_on_cpu is False)
                emb = batch_embeddings[i, :length_i].clone()

                if aggregate:
                    emb = emb.mean(dim=0)

                if output_directory is not None:
                    torch.save(emb, os.path.join(output_directory, f"{name}.pt"))
                else:
                    assert embedding_dict is not None
                    embedding_dict[name] = emb

            del batch_embeddings, sequence_tensor, attention_mask
            if (
                free_cuda_cache
                and torch.cuda.is_available()
                and torch.device(device).type == "cuda"
            ):
                torch.cuda.empty_cache()

    return embedding_dict


def ensemble_encoder_backend(
    ensemble,
    device,
    batch_size,
    model_manager=model_manager,
    encoder_path=None,
    ddpm_path=None,
):
    """
    Encode a stack of distance maps into the VAE's latent space.

    NOTE - this is the backend for starling.frontend.ensemble_generation.ensemble_encoder,
    which is the user-facing function.

    Parameters
    ----------
    ensemble : np.ndarray
        Distance maps in Angstroms, shape (n_conformers, L, L) with L <= 384.
        Maps smaller than 384 x 384 are zero-padded to that size, and the
        maps are converted to float32, the precision the VAE runs in.
    device : torch.device or str
        Device to run the encoder on.
    batch_size : int
        Number of distance maps encoded at once.
    model_manager : ModelManager
        Model manager instance
    encoder_path : str, optional
        Custom encoder path
    ddpm_path : str, optional
        Custom diffusion model path

    Returns
    -------
    np.ndarray
        Latents (the mode of the VAE's posterior), shape (n_conformers, 24, 24).
    """
    encoder_model, diffusion = model_manager.get_models(device=device, encoder_path=encoder_path, ddpm_path=ddpm_path)

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

    # the VAE's weights are float32, so float64 maps (numpy's default) would crash it
    ensemble = torch.from_numpy(ensemble.astype(np.float32, copy=False))
    ensemble = ensemble.unsqueeze(1)  # Add a channel dimension
    # inference_mode, or every batch builds an autograd graph through the
    # encoder, which multiplies its GPU memory several-fold
    with torch.inference_mode(), float32_matmul_precision(MODEL_FLOAT32_MATMUL_PRECISION):
        latent_spaces = [
            encoder_model.encode(ensemble[start : start + batch_size].to(device))
            .mode()
            .squeeze(1)
            .cpu()
            .numpy()
            for start in range(0, len(ensemble), batch_size)
        ]
    return np.concatenate(latent_spaces, axis=0)


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
        An initialized sampler (any ``starling.samplers.BaseSampler``).

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
    real_batch_count = (conformations + batch_size - 1) // batch_size
    starling_dm = []
    for batch, start in enumerate(range(0, conformations, batch_size), start=1):
        distance_maps = sampler.sample(
            min(batch_size, conformations - start),
            sequence=sequence,
            show_per_step_progress_bar=show_per_step_progress_bar,
            batch_count=batch,
            max_batch_count=real_batch_count,
            constraint=constraint,
        )
        maps = distance_maps[:, 0, : len(sequence), : len(sequence)]
        starling_dm.append(symmetrize_tensor_distance_maps(maps).cpu())
    return torch.cat(starling_dm, dim=0)


def _relaxation_seed_sequence(seed):
    """
    Build the seed sequence that every relaxation in one generate() call draws from.

    Relaxation (Langevin thermalization) gets its own random numbers rather
    than drawing from torch's global generator. Otherwise relaxing one
    sequence would shift the noise the sampler draws for the next one, so
    turning relaxation on or off would change every later distance map.

    Parameters
    ----------
    seed : int or None
        The seed passed to generate(). If None, torch.initial_seed() is
        used instead, so torch.manual_seed() upstream still makes
        relaxation reproducible (and an unseeded process gets a random
        seed, as before).

    Returns
    -------
    numpy.random.SeedSequence
        Seed sequence to pass to :func:`_next_relaxation_seed`.
    """
    return np.random.SeedSequence(torch.initial_seed() if seed is None else seed)


def _next_relaxation_seed(relaxation_seeds):
    """
    Draw the seed for one call to relax_conformations().

    Each call spawns a new, independent child of ``relaxation_seeds``, so
    every relaxation (each sequence, and each error-filter round) gets its
    own reproducible stream of random numbers.

    Parameters
    ----------
    relaxation_seeds : numpy.random.SeedSequence
        Seed sequence from :func:`_relaxation_seed_sequence`.

    Returns
    -------
    int
        A 32-bit seed.
    """
    (child,) = relaxation_seeds.spawn(1)
    return int(child.generate_state(1)[0])


def _relax_coordinates(
    sequence,
    coordinates,
    distance_maps,
    ionic_strength,
    device,
    batch_size,
    show_progress_bar,
    verbose=False,
    compile_forces=False,
    seed=None,
):
    """
    Relax MDS-reconstructed conformers with Mpipi-GG.

    Thin wrapper around :func:`starling.minimizer.relax_conformations` that
    accepts the Angstrom coordinates returned by MDS. Each
    conformer is restrained to the STARLING distance map it was built from, so
    local geometry (bond lengths, overlapping beads) is repaired while the
    global shape is kept.

    Parameters
    ----------
    sequence : str
        The amino acid sequence.

    coordinates : np.ndarray or torch.Tensor
        MDS coordinates in Angstroms, shape ``(n_conformers, L, 3)``.

    distance_maps : np.ndarray or torch.Tensor
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

    compile_forces : bool
        If True, fuse the CUDA thermalization forces (see
        :func:`starling.minimizer.relax_conformations`). This adds compilation
        startup time but speeds up sustained runs; it does not change the
        precision, force field or number of thermalization steps. Default False.

    seed : int or None
        Seed for the thermalization noise (see
        :func:`_next_relaxation_seed`). None draws from torch's global
        generator. Default None.

    Returns
    -------
    np.ndarray
        Relaxed coordinates in nanometres, shape ``(n_conformers, L, 3)``.
    """
    if not isinstance(coordinates, torch.Tensor):
        coordinates = np.asarray(coordinates)
    result = relax_conformations(
        coordinates,
        sequence,
        reference_distances=distance_maps,
        ionic_strength=ionic_strength,
        batch_size=batch_size,
        device=device,
        progress_bar=show_progress_bar,
        compile_forces=compile_forces,
        seed=seed,
    )

    if verbose:
        print(result.summary())

    return result.coordinates / configs.CONVERT_ANGSTROM_TO_NM


def _smacof_settings(relax):
    """
    How SMACOF starts, and how long it runs, before a given relaxation mode.

    Without relaxation nothing afterwards repairs what a shorter SMACOF run
    leaves behind (compressed bonds in particular), so relax=False uses the
    standard settings.

    Parameters
    ----------
    relax : str or bool
        "standard", "fast", or False for no relaxation.

    Returns
    -------
    tuple
        ``(initialization, n_iter)``: the classical-MDS initialization (see
        starling.structure.coordinates.MDS_INITIALIZATIONS) and SMACOF's
        maximum number of iterations.

    Raises
    ------
    ValueError
        If relax is not one of RELAX_MODES or False.
    """
    if relax == RELAX_FAST:
        return SUBSPACE_ITERATION, FAST_RELAX_SMACOF_ITERATIONS
    if relax is False or relax == RELAX_STANDARD:
        return FULL_EIGENDECOMPOSITION, DEFAULT_SMACOF_ITERATIONS
    raise ValueError(f"relax must be one of {RELAX_MODES} or False; got {relax!r}")


def _report_smacof_cap(stress_history):
    """
    Print how many conformations ran all of SMACOF's iterations.

    Parameters
    ----------
    stress_history : torch.Tensor
        SMACOF's stress history, shape (n_conformations, n_iter).
    """
    n_capped = int((~smacof_converged(stress_history)).sum())
    print(
        f"SMACOF: {n_capped} of {len(stress_history)} conformation(s) ran all "
        f"{stress_history.shape[1]} iterations without their stress settling (this is usual)"
    )


def _reconstruct_coordinates(
    sequence,
    distance_maps,
    ionic_strength,
    device,
    batch_size,
    show_progress_bar,
    relax,
    verbose,
    compile_forces=False,
    relaxation_seed=None,
):
    """
    Rebuild 3D structures from distance maps and, unless relax is False, relax them.

    Structures are rebuilt by weighted SMACOF, with the settings
    :func:`_smacof_settings` picks for the relaxation mode, then relaxed
    with :func:`_relax_coordinates`, without an intermediate CPU round trip.
    ``relax`` is "standard", "fast" or False. ``relaxation_seed`` (an int
    from :func:`_next_relaxation_seed`, or None for torch's global
    generator) seeds the thermalization noise when relaxing; the other
    parameters are as for :func:`_relax_coordinates`. With ``verbose``, also
    reports how many conformations SMACOF stopped early rather than at its
    iteration cap. Returns coordinates in nanometres.
    """
    initialization, n_iter = _smacof_settings(relax)
    coordinates, stress_history = _mds_coordinates(
        distance_maps,
        batch_size=batch_size,
        n_iter=n_iter,
        device=device,
        progress_bar=show_progress_bar,
        weights=map_error_weights(distance_maps.shape[-1], device=device),
        initialization=initialization,
    )

    if verbose:
        _report_smacof_cap(stress_history)

    if not relax:
        return coordinates.cpu().numpy() / configs.CONVERT_ANGSTROM_TO_NM

    return _relax_coordinates(
        sequence,
        coordinates,
        distance_maps,
        ionic_strength,
        device,
        batch_size,
        show_progress_bar,
        verbose,
        compile_forces=compile_forces,
        seed=relaxation_seed,
    )


def _generate_error_filtered_conformers(
    sampler,
    sequence,
    conformations,
    batch_size,
    show_per_step_progress_bar,
    constraint,
    return_structures,
    device,
    show_progress_bar,
    verbose=False,
    max_rounds=DEFAULT_MAX_ERROR_FILTER_ROUNDS,
    relax=False,
    ionic_strength=configs.DEFAULT_IONIC_STRENGTH,
    compile_forces=False,
    relaxation_seeds=None,
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

    If ``relax`` is "standard" or "fast", the reconstructed conformers are
    relaxed with Mpipi-GG *before* the trajectory-level screen. Relaxation repairs most
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

    show_progress_bar : bool
        Whether to show the reconstruction progress bar.

    verbose : bool
        If True, report how many conformers were discarded in each round.

    max_rounds : int
        Maximum number of sample-and-filter rounds before giving up. Guards
        against an unbounded loop if the model cannot produce clean conformers
        for a given sequence.

    relax : str or bool
        "standard" or "fast" (with ``return_structures`` True) relaxes the
        reconstructed conformers with Mpipi-GG before screening them, after
        the SMACOF settings :func:`_smacof_settings` picks for that mode.
        False skips relaxation. Default False.

    ionic_strength : float
        Ionic strength in mM used for the relaxation. Default is STARLING's
        default ionic strength.

    compile_forces : bool
        If True (and relaxing), fuse the CUDA thermalization forces used
        during relaxation; see :func:`_relax_coordinates`. Default False.

    relaxation_seeds : numpy.random.SeedSequence or None
        Where each round's relaxation seed comes from (see
        :func:`_next_relaxation_seed`). None builds one from
        torch.initial_seed(). Default None.

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
    if relaxation_seeds is None:
        relaxation_seeds = _relaxation_seed_sequence(None)

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
            coordinates = _reconstruct_coordinates(
                sequence,
                candidate_maps,
                ionic_strength,
                device,
                batch_size,
                show_progress_bar,
                relax,
                verbose,
                compile_forces=compile_forces,
                relaxation_seed=_next_relaxation_seed(relaxation_seeds),
            )

            ssprotein = SSTrajectory(TRJ=create_ca_topology_from_coords(sequence, coordinates)).proteinTrajectoryList[
                0
            ]

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
            f"Re-run without error filtering (remove_errors=False, or without "
            f"--remove-errors on the command line) to inspect the raw output."
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
    compile_forces=False,
    seed=None,
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

    conformations : int
        The number of conformations to generate for each sequence.

    device : torch.device or str
        The device to use for predictions.

    steps : int
        The number of denoising steps requested from the sampler. DDPM
        ignores this and always uses the full training schedule.

    sampler : str
        Name of the sampler to use, one of the keys of
        starling.samplers.SAMPLERS ('ddim', 'ddpm', 'dpmpp' or 'plms').

    return_structures : bool
        Whether to return the 3D structure.

    batch_size : int
        The batch size to use for sampling.

    output_directory : str or None
        If None, no output is saved.
        If not None, will save the output to the specified path.
        Each ensemble (distance maps, plus structures if
        return_structures=True) is saved as <sequence_name>.starling, and
        if return_structures=True the structures are also saved as
        <sequence_name>_STARLING.pdb and <sequence_name>_STARLING.xtc (or
        as a multi-model <sequence_name>_STARLING.pdb alone if
        pdb_trajectory=True).

    return_data : bool
        If True, the ensembles are returned in the output dictionary
        regardless of the output_directory. If False, the dictionary is
        returned empty; the reason to do this is that if you're predicting
        a large set of sequences it saves memory.

    verbose : bool
        Whether to print verbose output. Default is False.

    show_progress_bar : bool
        Whether to show a progress bar. Default is True.

    show_per_step_progress_bar : bool, optional
        whether to show progress bar per step.

    pdb_trajectory: bool
        If True (and structures are being saved), save the trajectory as a
        single multi-model PDB file instead of a PDB topology plus an XTC
        trajectory. Default is False.

    model_manager : ModelManager
        A ModelManager object to manage loaded models.
        This lets us avoid loading the model iteratively
        when calling generate multiple times in a single
        session. Default is model_manager, which is initialized
        outside of this function code block. To update the path
        to the models, update the paths in config.py, which are
        read into the ModelManager object located the
        model_loading.py

    ionic_strength : float, optional
        Ionic strength in mM used to condition the model (and for the
        relaxation, if relaxing). Default is 150.

    constraint : Constraint or None, optional
        A starling.inference.constraints.Constraint used to guide sampling.
        Default is None (no guidance).

    encoder_path : str, optional
        Path to a custom encoder model checkpoint file to use instead of the default.
        Default is None, which uses the default model path from configs.py.

    ddpm_path : str, optional
        Path to a custom diffusion model checkpoint file to use instead of the default.
        Default is None, which uses the default model path from configs.py.

    remove_errors : bool, optional
        If True, conformations with physically impossible distances (and,
        with return_structures=True, broken reconstructed structures) are
        discarded and replaced, so exactly ``conformations`` clean
        conformations are returned; see
        :func:`_generate_error_filtered_conformers`. Default is False.

    max_error_filter_rounds : int, optional
        Maximum number of sample-and-filter rounds when remove_errors is
        True before giving up with a RuntimeError. Default is
        DEFAULT_MAX_ERROR_FILTER_ROUNDS (10).

    relax : str or bool, optional
        "standard" or "fast": if return_structures is True, the
        MDS-reconstructed 3D structures are relaxed with Mpipi-GG while
        restrained to their STARLING distance maps (see starling.minimizer).
        This fixes bond lengths and overlapping beads while keeping global
        dimensions. The two modes relax identically; "fast" spends less
        time in SMACOF beforehand (see :func:`_smacof_settings`). False
        skips relaxation. Has no effect if return_structures is False.
        Default is False here; the user-facing generate() defaults to
        "standard".

    compile_forces : bool, optional
        If True, fuse the CUDA thermalization forces used during relaxation
        (requires relaxation and return_structures=True); see
        :func:`_relax_coordinates`. Default is False.

    seed : int or None, optional
        If an int, all of the sampler's noise comes from its own generator
        seeded with it. If None, the sampler draws from torch's global
        generator. Either way relaxation draws from separate seeds (see
        :func:`_relaxation_seed_sequence`). Default is None.

    Returns
    ---------------
    dict
        A dict with the sequence names as the key and
        a starling.structure.ensemble.Ensemble object for each
        sequence as values. The dict is empty if return_data is False.

        If output_directory is not none, the output will save to
        the specified path.
    """

    overall_start_time = time.time()

    # get models. This will only load once even if we call this
    # function multiple times.
    encoder_model, diffusion = model_manager.get_models(device=device, encoder_path=encoder_path, ddpm_path=ddpm_path)

    # the weights get_models() actually loaded, recorded in each ensemble's
    # metadata; configs.DEFAULT_*_WEIGHTS_PATH can differ from these if it was
    # changed after the model manager was imported
    encoder_reference, ddpm_reference = weights_references(encoder_path, ddpm_path)

    # Construct a sampler (DDPM ignores steps and always uses every timestep)
    sampler_class = SAMPLERS.get(sampler.lower())
    if sampler_class is None:
        raise ValueError(f"Error: sampler must be one of {', '.join(map(repr, SAMPLERS))}. Got {sampler}.")
    print(f"Using {sampler_class.name} sampler")

    # With a seed the sampler draws from its own generator, so the result
    # does not depend on what else the process has drawn and the global
    # random state is left alone. Without one it draws from torch's global
    # generator, as STARLING always has.
    sampler_generator = None
    if seed is not None:
        sampler_generator = torch.Generator(device=diffusion.device).manual_seed(seed)
    relaxation_seeds = _relaxation_seed_sequence(seed)

    sampler = sampler_class(
        ddpm_model=diffusion,
        encoder_model=encoder_model,
        n_steps=steps,
        ionic_strength=ionic_strength,
        generator=sampler_generator,
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
                show_progress_bar,
                verbose=verbose,
                max_rounds=max_error_filter_rounds,
                relax=relax,
                ionic_strength=ionic_strength,
                compile_forces=compile_forces,
                relaxation_seeds=relaxation_seeds,
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
                coordinates = _reconstruct_coordinates(
                    sequence,
                    sym_distance_maps,
                    ionic_strength,
                    device,
                    batch_size,
                    show_progress_bar,
                    relax,
                    verbose,
                    compile_forces=compile_forces,
                    relaxation_seed=_next_relaxation_seed(relaxation_seeds),
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

        E = Ensemble(
            final_distance_maps,
            sequence,
            ssprot_ensemble=ssprotein,
            ionic_strength=ionic_strength,
        )
        E.metadata["DEFAULT_ENCODER_WEIGHTS_PATH"] = encoder_reference
        E.metadata["DEFAULT_DDPM_WEIGHTS_PATH"] = ddpm_reference

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
                    filename_prefix=os.path.join(output_directory, seq_name + "_STARLING"),
                    pdb_trajectory=pdb_trajectory,
                )

            # save full ensemble. The extension is given explicitly because
            # save() strips whatever follows the last ".", which would otherwise
            # turn e.g. NP_000537.3 and NP_000537.4 into the same NP_000537.starling
            E.save(os.path.join(output_directory, f"{seq_name}.starling"), verbose=False)

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
            elapsed_time_structure_generation = end_time_structure_generation - start_time_structure_generation
            elapsed_time_prediction = end_time_prediction - start_time_prediction
            total_time = elapsed_time_structure_generation + elapsed_time_prediction

            print(f"\n\n##### SUMMARY OF SEQUENCE PREDICTION ({num + 1}/{len(sequence_dict)}) #####")
            print(f"Sequence name                       : {seq_name}")
            print(f"Sequence length                     : {len(sequence)}")
            print(f"Number of conformers                : {n_conformers}")
            print(f"Number of denoising steps           : {sampler.n_steps} ({sampler.name}; {steps} requested)")
            if remove_errors:
                # sampling, reconstruction/relaxation and error screening are
                # interleaved round by round, so they can only be timed together
                print(f"Total time for sampling, structures and error filtering : {round(total_time, 2)}s")
            else:
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
        print(f"\nTotal time (all sequences, all I/O) : {total_hours} hrs {total_minutes} mins {total_seconds} secs")

        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        print("STARLING predictions completed at:", formatted_datetime)
        print("")

    return output_dict
