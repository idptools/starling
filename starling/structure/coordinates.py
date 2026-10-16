import mdtraj as md
import numpy as np
import torch
from scipy.spatial import distance_matrix
from tqdm.auto import tqdm

from starling import configs
from starling.structure.weighted_stress import map_error_weights
from starling.utilities import check_device, float32_matmul_precision

# float32 matmul precision SMACOF always runs at, whatever the caller has set.
# SMACOF lands on a measurably different solution for about a third of
# conformations under TF32 ("high"), so pinning it means structures rebuilt
# from saved distance maps are the same in any process.
SMACOF_FLOAT32_MATMUL_PRECISION = "highest"

# SMACOF stopping rule. Both values were chosen empirically and are known to
# work for highly compact through highly expanded conformers at every length;
# starling/tests/test_smacof_envelope.py checks that across lengths and
# length-normalized ensemble dimensions, and any change to these (or to the
# starting point or weights) must keep passing it.
#
# A map counts as converged once its stress changes by less than
# DEFAULT_SMACOF_TOLERANCE (A^2) between iterations. The stress is a float32
# sum over all residue pairs (about 1e3 A^2 at 72 residues, 1e4-1e6 A^2 at
# 360), where adjacent float32 values are 6e-5 to 0.06 A^2 apart, so in
# practice this only fires when the stress stops changing altogether. On
# STARLING maps about 95% of conformations run all DEFAULT_SMACOF_ITERATIONS
# iterations (see smacof_converged). Most of those sit in a flat valley,
# where further iterations move them about an Angstrom without improving
# the fit; a few are still improving at the cap.
DEFAULT_SMACOF_ITERATIONS = 300
DEFAULT_SMACOF_TOLERANCE = 1e-4

# How the classical-MDS starting point finds the three largest eigenpairs of
# each map's Gram matrix. "full" diagonalizes the whole matrix with
# torch.linalg.eigh, which computes all n eigenpairs to keep three; it is the
# original method and the default. "subspace" finds only the few it needs by
# block subspace iteration (see _top_eigenpairs_by_subspace_iteration), which
# is about 10x faster at 380 residues on a GPU (3x at 100 residues).
# generate(relax="fast") uses it.
FULL_EIGENDECOMPOSITION = "full"
SUBSPACE_ITERATION = "subspace"
MDS_INITIALIZATIONS = (FULL_EIGENDECOMPOSITION, SUBSPACE_ITERATION)

# Block subspace iteration settings, chosen empirically. Each iteration
# shrinks the error in the three modes kept by the ratio of the 9th largest
# eigenvalue in magnitude to the 3rd largest positive one. The block carries
# more vectors than the three kept because STARLING maps are not quite
# Euclidean, and their Gram matrices have negative eigenvalues up to ~0.7 of
# the 3rd positive one in magnitude, which take places in the block. On the
# maps in test_smacof_envelope.py (20-380 residues, compact to expanded) the
# ratio was at most 0.36, so 20 iterations leave an error far below float32
# rounding; the starting structures matched the full eigendecomposition's to
# within 1.5e-4 of the largest distance. 20 rather than 12 iterations, for
# about 25% more time, also covers maps with flatter spectra (a noisy
# random-walk chain was 4e-3 off after 12 iterations and under 1e-3 after 20).
SUBSPACE_BLOCK_SIZE = 8
SUBSPACE_ITERATIONS = 20

# Seed for the random block that subspace iteration starts from. The block is
# drawn on the CPU, so it is the same on every device and every machine.
SUBSPACE_START_SEED = 0


def distance_matrix_to_3d_structure_torch_mds(
    target_distances,
    batch_size=100,
    n_iter=DEFAULT_SMACOF_ITERATIONS,
    tol=DEFAULT_SMACOF_TOLERANCE,
    device=None,
    progress_bar=True,
    weights=None,
    initialization=FULL_EIGENDECOMPOSITION,
):
    """
    SMACOF implementation using PyTorch with support for batched processing.

    On MPS, initialization and the weighted Laplacian pseudoinverse run on
    CPU in float32; SMACOF refinement runs on the requested device.

    This is the coordinate reconstruction path for CPU, CUDA, and MPS.
    Classical MDS provides a deterministic three-dimensional initialization;
    SMACOF then refines it against the requested weighted distance stress.
    Every float32 matmul runs at full precision (no TF32), whatever the
    caller's torch.set_float32_matmul_precision() setting, so the same maps
    give the same structures in any process.

    Parameters:
    -----------
    target_distances: numpy.ndarray or torch.Tensor
        tensor of shape (total_samples, n_points, n_points)
        containing distance maps in angstrom.

    batch_size: int
        size of each processing batch.

    n_iter: int
        Maximum number of iterations. Default 300. Most STARLING maps use
        all of them (see DEFAULT_SMACOF_TOLERANCE and smacof_converged).

    tol: float
        A map stops once its float32 stress changes by less than this (in
        A^2) between iterations. Default 1e-4, which is below float32
        resolution for all but the shortest chains, so in practice a map
        stops early only when its stress stops changing altogether.

    device: str, torch.device or None
        Device to use for computation. None (the default) picks CUDA, then
        MPS, then CPU, whichever is available first.

    progress_bar: bool
        Whether to display a progress bar.

    weights: torch.Tensor, optional
        Symmetric per-pair weights for the SMACOF stress. ``None`` retains
        ordinary unweighted stress.

    initialization: str
        How the classical-MDS starting point is computed: "full" (the
        default) diagonalizes each map's Gram matrix completely, while
        "subspace" finds only the three modes it needs by subspace
        iteration, which is much faster for long chains and gives
        essentially the same start. See MDS_INITIALIZATIONS.

    Returns:
    --------
    tuple

        [0] numpy.ndarray; Coordinates in angstrom, shape (total_samples, n_points, 3).
        [1] numpy.ndarray; Stress history (weighted stress, A^2), shape
        (total_samples, n_iter). Once a map stops, its stress is repeated
        for the remaining iterations. Pass it to smacof_converged() to see
        which maps stopped early rather than running all n_iter iterations.

    Raises:
    -------
    ValueError
        If any distance map contains NaN or infinite values, or
        initialization is not one of MDS_INITIALIZATIONS.

    """
    coordinates, stress = _mds_coordinates(
        target_distances, batch_size, n_iter, tol, device, progress_bar, weights, initialization
    )
    return coordinates.cpu().numpy(), stress.cpu().numpy()


def _classical_mds(distances, dim=3, initialization=FULL_EIGENDECOMPOSITION):
    """
    Starting coordinates for SMACOF by classical (Torgerson) MDS.

    The coordinates along each axis are an eigenvector of the map's
    double-centered Gram matrix, scaled by the square root of its
    eigenvalue, using the dim largest modes (Torgerson, Psychometrika 17,
    401-419, 1952). Negative eigenvalues, which non-Euclidean maps can
    give, are treated as zero.

    Parameters
    ----------
    distances : torch.Tensor
        Distance maps in Angstroms, shape (n_maps, n_points, n_points).
    dim : int
        Number of spatial dimensions. Default 3.
    initialization : str
        How the largest eigenpairs are found: "full" or "subspace" (see
        MDS_INITIALIZATIONS). Default "full".

    Returns
    -------
    torch.Tensor
        Coordinates in Angstroms, shape (n_maps, n_points, dim). With fewer
        than dim points, the missing axes are zero.
    """
    gram = _double_centered_gram(distances)
    n_modes = min(dim, distances.shape[-1])
    values, vectors = _top_eigenpairs(gram, n_modes, initialization)
    coordinates = vectors * values.clamp_min(0).sqrt()[:, None, :]
    return torch.nn.functional.pad(coordinates, (0, dim - n_modes))


def _double_centered_gram(distances):
    """
    Gram (inner product) matrix of the points implied by each distance map.

    Parameters
    ----------
    distances : torch.Tensor
        Distance maps in Angstroms, shape (n_maps, n_points, n_points).

    Returns
    -------
    torch.Tensor
        Gram matrices B = -J D^2 J / 2, where J is the centering matrix, in
        A^2 and shape (n_maps, n_points, n_points). For exactly embeddable
        maps this is X X^T for the centered coordinates X.
    """
    squared = distances.square()
    return -0.5 * (
        squared
        - squared.mean(dim=1, keepdim=True)
        - squared.mean(dim=2, keepdim=True)
        + squared.mean(dim=(1, 2), keepdim=True)
    )


def _top_eigenpairs(gram, n_modes, initialization):
    """
    The n_modes largest eigenpairs of each Gram matrix, by the requested method.

    Parameters
    ----------
    gram : torch.Tensor
        Symmetric matrices, shape (n_maps, n_points, n_points).
    n_modes : int
        Number of eigenpairs to return, at most n_points.
    initialization : str
        FULL_EIGENDECOMPOSITION or SUBSPACE_ITERATION.

    Returns
    -------
    tuple of torch.Tensor
        Eigenvalues, shape (n_maps, n_modes), and eigenvectors as columns,
        shape (n_maps, n_points, n_modes), both in ascending order of
        eigenvalue.

    Raises
    ------
    ValueError
        If initialization is not one of MDS_INITIALIZATIONS.
    """
    if initialization == FULL_EIGENDECOMPOSITION:
        return _top_eigenpairs_by_full_eigendecomposition(gram, n_modes)
    if initialization == SUBSPACE_ITERATION:
        return _top_eigenpairs_by_subspace_iteration(gram, n_modes)
    raise ValueError(f"initialization must be one of {MDS_INITIALIZATIONS}; got {initialization!r}")


def _top_eigenpairs_by_full_eigendecomposition(gram, n_modes):
    """
    The n_modes largest eigenpairs of each Gram matrix, from a full eigendecomposition.

    Parameters
    ----------
    gram : torch.Tensor
        Symmetric matrices, shape (n_maps, n_points, n_points).
    n_modes : int
        Number of eigenpairs to return, at most n_points.

    Returns
    -------
    tuple of torch.Tensor
        Eigenvalues, shape (n_maps, n_modes), and eigenvectors as columns,
        shape (n_maps, n_points, n_modes), both in ascending order of
        eigenvalue.
    """
    values, vectors = torch.linalg.eigh(gram)
    return values[:, -n_modes:], vectors[:, :, -n_modes:]


def _top_eigenpairs_by_subspace_iteration(gram, n_modes):
    """
    The n_modes largest eigenpairs of each Gram matrix, by block subspace iteration.

    Starting from a fixed random block of SUBSPACE_BLOCK_SIZE vectors, each
    iteration multiplies the block by the Gram matrix and re-orthonormalizes
    it, which turns it towards the eigenvectors with the largest eigenvalues.
    The Gram matrix is then diagonalized within the block (Rayleigh-Ritz),
    and the n_modes largest pairs kept. This is orthogonal iteration (Golub
    and Van Loan, Matrix Computations, 4th edition, chapter 8). It costs a
    few matrix products against the n^3 work of a full eigendecomposition.

    The block turns towards the eigenvalues largest in magnitude, negative
    ones included, so the three largest positive eigenpairs are found only
    if they are among the SUBSPACE_BLOCK_SIZE largest in magnitude, and they
    converge by the ratio of the next largest in magnitude to the third
    positive one. Non-Euclidean maps have negative eigenvalues; for STARLING
    maps these are at most ~0.7 of the third positive one, well inside the
    block (see SUBSPACE_ITERATIONS), and test_smacof_envelope.py checks that
    the result matches the full eigendecomposition's.

    Parameters
    ----------
    gram : torch.Tensor
        Symmetric positive semi-definite (up to non-Euclidean noise)
        matrices, shape (n_maps, n_points, n_points).
    n_modes : int
        Number of eigenpairs to return, at most n_points.

    Returns
    -------
    tuple of torch.Tensor
        Eigenvalues, shape (n_maps, n_modes), and eigenvectors as columns,
        shape (n_maps, n_points, n_modes), both in ascending order of
        eigenvalue.
    """
    n_maps, n_points, _ = gram.shape
    block_size = min(SUBSPACE_BLOCK_SIZE, n_points)

    # every map starts from the same block, drawn on the CPU so the start
    # (and so the result) does not depend on the device
    generator = torch.Generator().manual_seed(SUBSPACE_START_SEED)
    start_block = torch.randn(n_points, block_size, dtype=gram.dtype, generator=generator)
    basis = start_block.to(gram.device).expand(n_maps, n_points, block_size)

    for _ in range(SUBSPACE_ITERATIONS):
        basis, _ = torch.linalg.qr(torch.bmm(gram, basis))

    # Rayleigh-Ritz: the eigenpairs of the Gram matrix restricted to the block
    projected_gram = basis.transpose(1, 2) @ gram @ basis
    values, block_vectors = torch.linalg.eigh(projected_gram)
    vectors = basis @ block_vectors
    return values[:, -n_modes:], vectors[:, :, -n_modes:]


@float32_matmul_precision(SMACOF_FLOAT32_MATMUL_PRECISION)
def _mds_coordinates(
    target_distances,
    batch_size=100,
    n_iter=DEFAULT_SMACOF_ITERATIONS,
    tol=DEFAULT_SMACOF_TOLERANCE,
    device=None,
    progress_bar=True,
    weights=None,
    initialization=FULL_EIGENDECOMPOSITION,
):
    """Return coordinates (Angstroms) and stress tensors on the requested device."""
    device = check_device(device)

    if initialization not in MDS_INITIALIZATIONS:
        raise ValueError(f"initialization must be one of {MDS_INITIALIZATIONS}; got {initialization!r}")

    target_distances = torch.as_tensor(target_distances)

    # A non-finite map makes CUDA's eigensolver fail for its whole batch with
    # a misleading "ill-conditioned" error, while CPU silently returns NaN
    # coordinates, so refuse it up front on every device
    finite_frames = torch.isfinite(target_distances).flatten(start_dim=1).all(dim=1)
    if not finite_frames.all():
        bad_frames = torch.nonzero(~finite_frames).flatten().tolist()
        raise ValueError(
            f"{len(bad_frames)} of {len(finite_frames)} distance map(s) contain NaN or infinite "
            f"values and cannot be reconstructed (frame indices: {bad_frames})"
        )

    total_samples = target_distances.shape[0]
    n_points = target_distances.shape[1]
    dim = 3
    eps = 1e-12

    laplacian_pinv = None
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        if weights.shape != (n_points, n_points):
            raise ValueError("weights must have shape (n_points, n_points)")
        if (
            not torch.isfinite(weights).all()
            or torch.any(weights < 0)
            or not torch.allclose(weights, weights.T)
            or torch.any(weights.diagonal() != 0)
        ):
            raise ValueError("weights must be finite, symmetric, nonnegative, and hollow")
        laplacian = torch.diag(weights.sum(dim=1)) - weights
        # MPS does not support the SVD used by pinv reliably.
        laplacian_pinv = torch.linalg.pinv(laplacian.cpu() if device.type == "mps" else laplacian).to(device)

    X_results = []
    stress_results = []

    for start in tqdm(range(0, total_samples, batch_size), disable=not progress_bar):
        end = min(start + batch_size, total_samples)
        batch_distances = target_distances[start:end].to(device=device, dtype=torch.float32)

        # Classical MDS gives the starting point. MPS lacks the eigensolver
        # (and QR) it needs, so on MPS it runs on the CPU.
        initial_distances = batch_distances.cpu() if device.type == "mps" else batch_distances
        X = _classical_mds(initial_distances, dim, initialization).to(device=device)
        X = X - X.mean(dim=1, keepdim=True)
        weighted_distances = batch_distances if weights is None else weights * batch_distances

        # Initialize stress tracking
        stress_history = torch.zeros(end - start, n_iter, dtype=torch.float32, device=device)
        old_stress = torch.full((end - start,), float("inf"), dtype=torch.float32, device=device)
        converged = torch.zeros(end - start, dtype=torch.bool, device=device)

        for it in range(n_iter):
            diff = X.unsqueeze(2) - X.unsqueeze(1)
            D = torch.norm(diff, dim=3) + eps

            residual = (D - batch_distances) ** 2
            stress = torch.sum(residual if weights is None else weights * residual, dim=(1, 2))
            stress_history[:, it] = stress

            converged = converged | _stress_has_settled(old_stress, stress, tol)
            # Freeze each converged map immediately; inspect batch completion
            # less often on GPUs to avoid a host synchronization every step.
            if (device.type == "cpu" or (it + 1) % 8 == 0) and torch.all(converged):
                # Keep the documented fixed width when batches stop independently.
                stress_history[:, it + 1 :] = stress[:, None]
                break

            B = torch.where(D > eps, -weighted_distances / D, 0.0)

            row_sums = B.sum(dim=2)
            B.diagonal(dim1=1, dim2=2).copy_(-row_sums)

            B_X = torch.bmm(B, X)
            if weights is None:
                X_new = B_X / n_points
            else:
                assert laplacian_pinv is not None
                X_new = torch.matmul(laplacian_pinv, B_X)

            X = torch.where(converged[:, None, None], X, X_new - X_new.mean(dim=1, keepdim=True))

            old_stress = stress

        X_results.append(X)
        stress_results.append(stress_history)

    # Concatenate all chunk results
    X_final = torch.cat(X_results, dim=0)
    stress_final = torch.cat(stress_results, dim=0)

    return X_final, stress_final


def _stress_has_settled(old_stress, stress, tol):
    """
    SMACOF's stopping test: has each map's stress changed by less than tol?

    Shared by the SMACOF loop and smacof_converged(), so the two can never
    disagree about which maps stopped.

    Parameters
    ----------
    old_stress : torch.Tensor
        float32 stress at the previous iteration, in A^2.
    stress : torch.Tensor
        float32 stress at this iteration, same shape.
    tol : float
        Tolerance in A^2.

    Returns
    -------
    torch.Tensor
        Boolean tensor, True where the stress has settled.
    """
    return torch.abs(old_stress - stress) < tol


def smacof_converged(stress_history, tol=DEFAULT_SMACOF_TOLERANCE):
    """
    Report which maps SMACOF stopped early, rather than at its iteration cap.

    A map stops early once its stress settles (see DEFAULT_SMACOF_TOLERANCE).
    One that never does runs all n_iter iterations, and its structure is
    wherever SMACOF had got to by then. On STARLING maps that is the usual
    case (about 95% of conformations), so a False here is not an error.

    Parameters
    ----------
    stress_history : numpy.ndarray or torch.Tensor
        The stress history returned by
        distance_matrix_to_3d_structure_torch_mds(), shape
        (n_maps, n_iter).
    tol : float, optional
        The tolerance the reconstruction was run with. Default
        DEFAULT_SMACOF_TOLERANCE.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape (n_maps,), True for maps whose stress
        settled before the iteration cap.
    """
    history = torch.as_tensor(stress_history)
    settled = _stress_has_settled(history[:, :-1], history[:, 1:], tol)
    return settled.any(dim=1).cpu().numpy()


def compare_distance_matrices(original_distance_matrix, coords, return_abs_diff=True):
    """
    Function to compare the original distance matrix with the
    computed distance matrix.

    Parameters
    ----------
    original_distance_matrix : np.ndarray
        The original distance matrix.

    coords : np.ndarray
        The computed 3D coordinates.

    return_abs_diff : bool
        Whether to return the absolute difference between the
        original and computed distance matrices, or the signed
        difference (original - computed), by default True.

    Returns
    -------
    tuple
        A 2-element tuple of np.ndarray:

        - [0]: The distance matrix computed from the 3D coordinates.
        - [1]: The difference between the original and computed distance
          matrices: absolute if return_abs_diff is True (default), otherwise
          signed (original - computed).
    """

    # compute the redundant inter-residue distance map based on the
    # passed coordinates
    computed_distance_matrix = distance_matrix(coords, coords)

    # calculate the difference between the original distance map and the distance map
    # derived from the input 3D structure
    difference_matrix = original_distance_matrix - computed_distance_matrix

    if return_abs_diff:
        difference_matrix = np.abs(difference_matrix)

    # return the computed distance matrix and the difference matrix
    return computed_distance_matrix, difference_matrix


def create_ca_topology_from_coords(sequence, coords):
    """
    Creates a CA backbone topology from a protein sequence and 3D coordinates.

    Parameters
    ----------
    sequence : str
        Protein sequence as a string of amino acid single-letter codes.

    coords : np.ndarray
        3D coordinates for each CA atom in nanometers (MDTraj units), shape
        (n_frames, n_residues, 3) or (n_residues, 3) for a single frame.

    Returns
    ----------
    md.Trajectory
         MDTraj trajectory object containing the CA backbone topology and coordinates.
    """

    # Create an empty topology
    topology = md.Topology()

    # Add a chain to the topology
    chain = topology.add_chain()

    # -- topology construction loop

    for i, res in enumerate(sequence):
        try:
            res_three_letter = configs.AA_ONE_TO_THREE[res]
        except KeyError:
            raise ValueError(f"Invalid amino acid: {res}")

        residue = topology.add_residue(res_three_letter, chain)

        # Add a CA atom to the residue; added formal_charge=0 to avoid warnings/errors
        # about missing formal charges in MDTraj >1.10.0
        ca_atom = topology.add_atom("CA", md.element.carbon, residue, formal_charge=0)

        # Connect the CA atom to the previous CA atom (if not the first residue)
        if i > 0:
            topology.add_bond(topology.atom(i - 1), ca_atom)

    # --- end of topology construction loop

    # Ensure the coordinates are in the right shape (1, num_atoms, 3)
    if coords.ndim != 3:
        coords = coords[np.newaxis, :, :]

    # Create an MDTraj trajectory object with the topology and coordinates
    traj = md.Trajectory(coords, topology)

    return traj


# Function to save the MDTraj trajectory to a specified file
def save_trajectory(traj, filename):
    """
    Saves the MDTraj trajectory to a specified file. This invokes
    the trak.save() method of the MDTraj trajectory object.

    Parameters
    -----------
    traj : md.Trajectory
        The MDTraj trajectory object to save.

    filename : str
        The name of the file to save the trajectory .
    """

    traj.save(filename)


def generate_3d_coordinates_from_distances(device, batch_size, distance_maps, progress_bar=True):
    """
    Function to generate 3D coordinates from distance maps.

    This is the parent function which uses classical MDS initialization
    followed by weighted Torch SMACOF on the requested device.

    This function is called in:

        1. inference.generation, where the distance maps are
           tensors.

        2. From the Ensemble() object, where the distance maps
           are numpy arrays.

    Parameters
    ----------
    device : str
        The device to use for computation.

    batch_size : int
        The batch size for processing the distance maps.

    distance_maps : numpy.ndarray or torch.Tensor
        Either an nd.array or a pytorch tensor of distance
        maps. Distances are in angstrom.

    Returns
    -------
    numpy.ndarray
        Coordinates in nanometers, shape (total_samples, n_points, 3).
    """

    coordinates, _ = distance_matrix_to_3d_structure_torch_mds(
        distance_maps,
        batch_size=batch_size,
        device=device,
        progress_bar=progress_bar,
        weights=map_error_weights(distance_maps.shape[-1], device=device),
    )
    coordinates /= configs.CONVERT_ANGSTROM_TO_NM

    return coordinates
