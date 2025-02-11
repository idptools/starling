import os
import time

import mdtraj as md
import numpy as np
import torch
import torch.optim as optim
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS

from starling import configs

from tqdm import tqdm
from starling import utilities


def get_tensor_dtype(device):
    """
    Function which returns a tensor dtype based on
    the device passed. This is to support the fact that
    mps does not work with float64 tensors and will set
    them to be cast to float32. This is an internal
    function that really only ends up being relevant
    for gradien descent reconstruction.

    Parameters
    ---------------
    device : torch.device
        Device being used

    returns
    --------------------
    torch.dtype
        Returns the type

    """

    # as of 2024-11 mps does not support float64
    if str(device) == "mps":
        tensor_dtype = torch.float32
    else:
        tensor_dtype = torch.float64

    return tensor_dtype


def compute_pairwise_distances(coords):
    """Function to compute the pairwise distances in 3D space.

    Parameters
    ----------
    coords : torch.Tensor
        A tensor of shape (n, 3) containing the 3D coordinates of n points.

    Returns
    -------
    torch.Tensor
        A tensor of pairwise distances between the points in 3D space.
    """
    return torch.cdist(coords, coords)


def loss_function(original_distance_matrix, coords):
    """
    Function to compute the loss between the original distance
    matrix and the computed distance matrix.

    Parameters
    ----------
    original_distance_matrix : torch.Tensor
        A tensor of shape (n, n) containing the original pairwise
        distances between n points.

    coords : torch.Tensor
        A tensor of shape (n, 3) containing the 3D coordinates
        of n points.

    Returns
    -------
    torch.Tensor
        The mean squared error between the original and computed
        distance matrices, considering only the upper triangle.

    """
    computed_distances = compute_pairwise_distances(coords)

    # Create a mask for the upper triangle
    upper_triangle_mask = torch.triu(
        torch.ones_like(original_distance_matrix), diagonal=1
    ).bool()

    # Apply the mask to both the original and computed distance matrices
    masked_original = original_distance_matrix[upper_triangle_mask]
    masked_computed = computed_distances[upper_triangle_mask]

    # Compute the mean squared error only for the masked elements
    loss = torch.nn.functional.mse_loss(masked_computed, masked_original)

    return loss


def create_incremental_coordinates(n_points, distance, device):
    """
    TO DO: Add docstring
    """

    coordinates = torch.zeros(
        (n_points, 3), dtype=get_tensor_dtype(device), device=device
    )

    # Start the first coordinate at (0, 0, 0)
    for i in range(1, n_points):
        # Generate a random direction vector
        direction = torch.randn(3, dtype=get_tensor_dtype(device), device=device)
        direction /= torch.norm(direction)  # Normalize to get a unit vector

        # Calculate the new coordinate by adding the direction scaled by the distance
        new_coordinate = coordinates[i - 1] + direction * distance

        coordinates[i] = new_coordinate

    return torch.nn.Parameter(coordinates)


def distance_matrix_to_3d_structure_torch_mds(
    target_distances, batch_size=100, n_iter=300, tol=1e-4, device='cuda', progress_bar=True):
    """
    SMACOF implementation using PyTorch with support for batched processing. 
    
    NB; as of Feb 2024 this is substantially slower for MPS than the other approach, 
    because MPS seems to fail and fall back on CPU in this parallelized version. This 
    is 1.5-2x slower than the other approach. To keep this simple, there is a check in 
    the parent function generate_3d_coordinates_from_distances() that defaults to the 
    non-torch implementation if the device is MPS and we're on macOS <= 14. Hopefully 
    this is fixed in macOS15...

    Parameters:
    -----------
    target_distances: pytorch.Tensor
        tensor of shape (total_samples, n_points, n_points)

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
        
    Returns:
    --------
    tuple
    
        [0] pytorch.Tensor; A tensor of shape (total_samples, n_points, 3)
        [1] pytorch.Tensor; A tensor of shape (total_samples, n_iter)
        
    """

    # if we passed in a numpy array, convert it to a torch tensor
    # on the correct device        
    if isinstance(target_distances, np.ndarray):        
        target_distances = torch.from_numpy(target_distances).to(device=device)
    
    total_samples = target_distances.shape[0]
    n_points = target_distances.shape[1]
    dim = 3
    eps = 1e-12

    X_results = []
    stress_results = []

    # start the progress bar if requested
    if progress_bar == True:
        local_generator = tqdm(range(0, total_samples, batch_size))
    else:
        local_generator = range(0, total_samples, batch_size)


    #for start in range(0, total_samples, batch_size):
    for start in local_generator:
        end = min(start + batch_size, total_samples)
        batch_distances = target_distances[start:end].to(device)

        # Initialize random coordinates for the current batch_size
        X = torch.randn(end - start, n_points, dim, dtype=torch.float32, device=device)
        X = X - X.mean(dim=1, keepdim=True)

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

            stress = torch.sum((D - batch_distances) ** 2, dim=(1, 2))
            stress_history[:, it] = stress

            converged = converged | (torch.abs(old_stress - stress) < tol)
            if torch.all(converged):
                stress_history = stress_history[:, : it + 1]
                break

            B = torch.zeros_like(D)
            mask = D > eps

            # the offending line for macOS
            B[mask] = -batch_distances[mask] / D[mask]

            # this seems to work on MPS, whereas line above triggers an warning
            # and drops down to CPU; however need to test
            # B = torch.where(mask, -batch_distances / D, B)

            row_sums = B.sum(dim=2)
            B.diagonal(dim1=1, dim2=2).copy_(-row_sums)

            X_new = torch.bmm(B, X) / n_points
            X[~converged] = X_new[~converged]
            X[~converged] = X[~converged] - X[~converged].mean(dim=1, keepdim=True)

            old_stress = stress

        X_results.append(X.cpu())
        stress_results.append(stress_history.cpu())

    # close the progress bar if opened
    if progress_bar == True:
        local_generator.close()        

    # Concatenate all chunk results
    X_final = torch.cat(X_results, dim=0)
    stress_final = torch.cat(stress_results, dim=0)

    return X_final.numpy(), stress_final.numpy()


def distance_matrix_to_3d_structure_mds(distance_matrix, **kwargs):
    """
    Generate 3D coordinates from a distance matrix using
    multidimensional scaling (MDS).

    NB: by default the MDS object is initialized with
    the following defaults, although these can be overridden
    by passing them in the kwargs:

    n_components = 3
    dissimilarity = "precomputed"
    n_init = configs.DEFAULT_MDS_NUM_INIT  (default setting)
    n_jobs = configs.DEFAULT_CPU_COUNT_MDS (default setting)
    normalized_stress = 'auto'

    Parameters
    ----------
    distance_matrix : torch.Tensor
        A 2D tensor representing the distance matrix.

    kwargs : dict
        Keyword arguments to pass to scikit-learn's MDS
        algorithm.

    Returns
    -------
    torch.Tensor
        A 3D tensor representing the coordinates of the
        atoms.

    """    
    # Set the default values for n_init and n_jobs if not provided in kwargs
    # this matches the default values in scikit-learn's MDS
    n_init = kwargs.pop("n_init", configs.DEFAULT_MDS_NUM_INIT)
    n_jobs = kwargs.pop("n_jobs", configs.DEFAULT_CPU_COUNT_MDS)

    # Initialize MDS with 3 components (for 3D) and the specified parameters
    # nb: normalized_stress = 'auto' explicitly as this is the default
    # value in sci-kit learn >1.4 , but before that it was False, so this just
    # ensures version-independent behavior in the MDS call
    mds = MDS(
        n_components=3,
        dissimilarity="precomputed",
        n_init=n_init,
        n_jobs=n_jobs,
        normalized_stress="auto",
        **kwargs,
    )

    # Fit the MDS model to the distance matrix
    coords = mds.fit_transform(distance_matrix)

    return coords


def distance_matrix_to_3d_structure_gd(
    original_distance_matrix,
    num_iterations=5000,
    learning_rate=1e-3,
    device="cuda:0",
    verbose=True,
):
    """
    Function to reconstruct a 3D structure from a
    distance matrix using gradient descent.

    Parameters
    ----------
    original_distance_matrix : torch.Tensor or numpy.ndarray
        The original distance matrix.

    num_iterations : int, optional
        Number of iterations for gradient descent, by default 5000.

    learning_rate : float, optional
        Learning rate for the optimizer, by default 1e-3.

    device : str, optional
        Device to which tensors are moved, by default "cuda:0".

    verbose : bool, optional
        Whether to print progress, by default True.

    Returns
    -------
    numpy.ndarray
        The reconstructed 3D coordinates.

    NB: As of Nov 2024, Apple MPS does not support float64 tensors, so we cast
    tensors to float32 in the case of MPS being used. Note that this is actually
    slower than using CPU, but we provide support for this in case someone wants
    it and/or it gets faster in the future...

    """

    if isinstance(original_distance_matrix, torch.Tensor):
        original_distance_matrix = original_distance_matrix.to(
            device, dtype=get_tensor_dtype(device)
        )

    else:
        original_distance_matrix = torch.tensor(
            original_distance_matrix, dtype=get_tensor_dtype(device), device=device
        )

    coords = create_incremental_coordinates(
        original_distance_matrix.shape[0], 3.6, device=device
    )
    # coords = torch.randn((original_distance_matrix.size(0), 3), requires_grad=True, device=device,dtype=torch.float64)

    # optimizer = optim.SGD([coords], lr=learning_rate, momentum=0.99, nesterov=True)
    optimizer = optim.Adam([coords], lr=learning_rate)

    for i in range(num_iterations):
        optimizer.zero_grad()

        loss = loss_function(original_distance_matrix, coords)

        loss.backward()

        optimizer.step()

        if i % 100 == 0 and verbose:
            print(f"Iteration {i}, Loss: {loss.item()}")

    return coords.detach().cpu().numpy()


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
    tuple (np.ndarray, np.ndarray)
        [0] - The distance matrix computed from the 3D coordinates.
        [1] - The absolute difference between the original and computed
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

        # Add a CA atom to the residue
        ca_atom = topology.add_atom("CA", md.element.carbon, residue)

        # Connect the CA atom to the previous CA atom (if not the first residue)
        if i > 0:
            topology.add_bond(topology.atom(i - 1), ca_atom)

    # --- end of topology construction loop

    # Ensure the coordinates are in the right shape (1, num_atoms, 3)
    if coords.ndim != 3:
        coords = coords[np.newaxis, :, :]

    # commented out for now
    # else:
    #    print(coords.shape)

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


def generate_3d_coordinates_from_distances(device, 
                                           batch_size, 
                                           num_cpus_mds, 
                                           num_mds_init, 
                                           distance_maps, 
                                           progress_bar=True):
    """
    Function to generate 3D coordinates from distance maps.

    This is the parent function which will then choose either
    the fasts or most approriate method to use based on the
    requested device and the actual OS.

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
        The batch size for processing the distance maps if
        we use the torch implementation.

    num_cpus_mds : int
        The number of CPUs to use for MDS if we use the
        SciPy implementation.

    num_mds_init : int
        The number of initializations to use for MDS if
        we use the SciPy implementation.

    distance_maps : list
        Either an nd.array or a pytorch tensor of distance
        maps.
    """

    # cast device to string for ffs
    device = str(device)

    # leaving this in case we have to revert  
    # note; macOS <= 14  gives a "MPS: nonzero op" error, so because of
    # this we force 'cpu' here if on macOS 
    # macOS_version = utilities.get_macOS_version()

    #if (macOS_version > 0 and macOS_version < 15) or device == "cpu":       
    if device == 'cpu':  

        ## NB: It seemed like we needed to do this at one point, but
        ## maybe not anymore? Leaving here in case we need it later
        # if we passed in a numpy array, convert it to a torch tensor
        # on the correct device (note if we're here we're on CPU)
        #if isinstance(distance_maps, np.ndarray):
        #    distance_maps = torch.from_numpy(distance_maps).to('cpu')

        # open a progress bar if requested
        if progress_bar == True:
            dm_generator = tqdm(distance_maps)

        coordinates = (
            np.array(
                [
                    distance_matrix_to_3d_structure_mds(
                        dist_map,
                        n_jobs=num_cpus_mds,
                        n_init=num_mds_init,
                    )
                    for dist_map in dm_generator
                ]
            )
            / configs.CONVERT_ANGSTROM_TO_NM
        )

        # close the progress bar if opened
        if progress_bar == True:
            dm_generator.close()
    else:

        # call the torch MDS implementation
        coordinates, _ = distance_matrix_to_3d_structure_torch_mds(distance_maps, batch_size=batch_size, device=device, progress_bar=progress_bar)
        coordinates /= configs.CONVERT_ANGSTROM_TO_NM

    return coordinates
