import mdtraj as md
import numpy as np
import torch
from scipy.spatial import distance_matrix
from tqdm.auto import tqdm

from starling import configs
from starling.structure.weighted_stress import map_error_weights


def distance_matrix_to_3d_structure_torch_mds(
    target_distances,
    batch_size=100,
    n_iter=300,
    tol=1e-4,
    device="cuda",
    progress_bar=True,
    weights=None,
):
    """
    SMACOF implementation using PyTorch with support for batched processing.

    NB; as of Feb 2024 this is substantially slower for MPS than the other approach,
    because MPS seems to fail and fall back on CPU in this parallelized version. This
    is 1.5-2x slower than the other approach.

    This is the coordinate reconstruction path for CPU, CUDA, and MPS.
    Classical MDS provides a deterministic three-dimensional initialization;
    SMACOF then refines it against the requested weighted distance stress.

    Parameters:
    -----------
    target_distances: numpy.ndarray or torch.Tensor
        tensor of shape (total_samples, n_points, n_points)
        containing distance maps in angstrom.

    batch_size: int
        size of each processing batch.

    n_iter: int
        Maximum number of iterations.

    tol: float
        Convergence tolerance.

    device: str
        Device to use for computation.

    progress_bar: bool
        Whether to display a progress bar.

    weights: torch.Tensor, optional
        Symmetric per-pair weights for the SMACOF stress. ``None`` retains
        ordinary unweighted stress.

    Returns:
    --------
    tuple

        [0] numpy.ndarray; Coordinates in angstrom, shape (total_samples, n_points, 3).
        [1] numpy.ndarray; Stress history, shape (total_samples, n_iter).

    """
    device = torch.device(device)

    target_distances = torch.as_tensor(target_distances)

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
            raise ValueError(
                "weights must be finite, symmetric, nonnegative, and hollow"
            )
        laplacian = torch.diag(weights.sum(dim=1)) - weights
        # MPS does not support the SVD used by pinv reliably.
        laplacian_pinv = torch.linalg.pinv(
            laplacian.cpu() if device.type == "mps" else laplacian
        ).to(device)

    X_results = []
    stress_results = []

    for start in tqdm(range(0, total_samples, batch_size), disable=not progress_bar):
        end = min(start + batch_size, total_samples)
        batch_distances = target_distances[start:end].to(
            device=device, dtype=torch.float32
        )

        # Double-center squared distances and retain the three positive modes.
        # MPS lacks float64, so initialize on CPU there before returning to MPS.
        initial_distances = (
            batch_distances.cpu() if device.type == "mps" else batch_distances
        )
        squared = initial_distances.to(torch.float64).square()
        gram = -0.5 * (
            squared
            - squared.mean(dim=1, keepdim=True)
            - squared.mean(dim=2, keepdim=True)
            + squared.mean(dim=(1, 2), keepdim=True)
        )
        values, vectors = torch.linalg.eigh(gram)
        dimensions = min(dim, n_points)
        X = (
            vectors[:, :, -dimensions:]
            * values[:, -dimensions:].clamp_min(0).sqrt()[:, None, :]
        )
        X = torch.nn.functional.pad(X, (0, dim - dimensions)).to(
            device=device, dtype=torch.float32
        )
        X = X - X.mean(dim=1, keepdim=True)
        weighted_distances = (
            batch_distances if weights is None else weights * batch_distances
        )

        # Initialize stress tracking
        stress_history = torch.zeros(
            end - start, n_iter, dtype=torch.float32, device=device
        )
        old_stress = torch.full(
            (end - start,), float("inf"), dtype=torch.float32, device=device
        )
        converged = torch.zeros(end - start, dtype=torch.bool, device=device)

        for it in range(n_iter):
            diff = X.unsqueeze(2) - X.unsqueeze(1)
            D = torch.norm(diff, dim=3) + eps

            residual = (D - batch_distances) ** 2
            stress = torch.sum(
                residual if weights is None else weights * residual, dim=(1, 2)
            )
            stress_history[:, it] = stress

            converged = converged | (torch.abs(old_stress - stress) < tol)
            if torch.all(converged):
                # Keep the documented fixed width when batches stop independently.
                stress_history[:, it + 1 :] = stress[:, None]
                break

            B = torch.zeros_like(D)
            mask = D > eps

            B = torch.where(mask, -weighted_distances / D, B)

            row_sums = B.sum(dim=2)
            B.diagonal(dim1=1, dim2=2).copy_(-row_sums)

            B_X = torch.bmm(B, X)
            if weights is None:
                X_new = B_X / n_points
            else:
                assert laplacian_pinv is not None
                X_new = torch.matmul(laplacian_pinv, B_X)

            X = torch.where(converged.unsqueeze(1).unsqueeze(2), X, X_new)
            X = torch.where(
                converged.unsqueeze(1).unsqueeze(2), X, X - X.mean(dim=1, keepdim=True)
            )

            old_stress = stress

        X_results.append(X.cpu())
        stress_results.append(stress_history.cpu())

    # Concatenate all chunk results
    X_final = torch.cat(X_results, dim=0)
    stress_final = torch.cat(stress_results, dim=0)

    return X_final.numpy(), stress_final.numpy()


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
        - [1]: The absolute difference between the original and computed
          distance matrices.
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
        3D coordinates for each CA atom.

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


def generate_3d_coordinates_from_distances(
    device, batch_size, distance_maps, progress_bar=True
):
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
