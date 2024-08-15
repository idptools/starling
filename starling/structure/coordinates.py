import torch
import mdtraj as md
import torch.optim as optim
import numpy as np
from scipy.spatial import distance_matrix

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

#def loss_function(original_distance_matrix, coords):
#    """Function to compute the loss between the original distance matrix and the computed distance matrix.
#
#    Parameters
#    ----------
#    original_distance_matrix : torch.Tensor
#        A tensor of shape (n, n) containing the original pairwise distances between n points.
#    coords : torch.Tensor
#        A tensor of shape (n, 3) containing the 3D coordinates of n points.
#
#    Returns
#    -------
#    torch.Tensor
#        The mean squared error between the original and computed distance matrices.
#    """
#    computed_distances = compute_pairwise_distances(coords)
#    return torch.nn.functional.mse_loss(computed_distances, original_distance_matrix)


def loss_function(original_distance_matrix, coords):
    """Function to compute the loss between the original distance matrix and the computed distance matrix.

    Parameters
    ----------
    original_distance_matrix : torch.Tensor
        A tensor of shape (n, n) containing the original pairwise distances between n points.
    coords : torch.Tensor
        A tensor of shape (n, 3) containing the 3D coordinates of n points.

    Returns
    -------
    torch.Tensor
        The mean squared error between the original and computed distance matrices, considering only the upper triangle.
    """
    computed_distances = compute_pairwise_distances(coords)
    
    # Create a mask for the upper triangle
    upper_triangle_mask = torch.triu(torch.ones_like(original_distance_matrix), diagonal=1).bool()
    
    # Apply the mask to both the original and computed distance matrices
    masked_original = original_distance_matrix[upper_triangle_mask]
    masked_computed = computed_distances[upper_triangle_mask]

    # Compute the mean squared error only for the masked elements
    loss = torch.nn.functional.mse_loss(masked_computed, masked_original)

    return loss


def create_incremental_coordinates(n_points, distance, device):
    coordinates = torch.zeros((n_points, 3), dtype=torch.float64, device=device)
    # Start the first coordinate at (0, 0, 0)

    for i in range(1, n_points):
        # Generate a random direction vector
        direction = torch.randn(3, dtype=torch.float64, device=device)
        direction /= torch.norm(direction)  # Normalize to get a unit vector

        # Calculate the new coordinate by adding the direction scaled by the distance
        new_coordinate = coordinates[i - 1] + direction * distance

        coordinates[i] = new_coordinate

    return torch.nn.Parameter(coordinates)


def distance_matrix_to_3d_structure_gd(original_distance_matrix,
                                       num_iterations=5000,
                                       learning_rate=1e-3,
                                       device="cuda:0",
                                       verbose=True):
    """Function to reconstruct a 3D structure from a distance matrix using gradient descent.

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
    """
    if isinstance(original_distance_matrix, torch.Tensor):
        original_distance_matrix = original_distance_matrix.to(device, dtype=torch.float64)
    else:
        original_distance_matrix = torch.tensor(original_distance_matrix, dtype=torch.float64, device=device)

    coords = create_incremental_coordinates(original_distance_matrix.shape[0], 3.6, device=device)
    #coords = torch.randn((original_distance_matrix.size(0), 3), requires_grad=True, device=device,dtype=torch.float64)


    #optimizer = optim.SGD([coords], lr=learning_rate, momentum=0.99, nesterov=True)
    optimizer = optim.Adam([coords], lr=learning_rate)

    for i in range(num_iterations):
        optimizer.zero_grad()

        loss = loss_function(original_distance_matrix, coords)

        loss.backward()

        optimizer.step()

        if i % 100 == 0 and verbose:
            print(f"Iteration {i}, Loss: {loss.item()}")


    return coords.detach().cpu().numpy()

def compare_distance_matrices(original_distance_matrix, coords):
    computed_distance_matrix = distance_matrix(coords, coords)
    difference_matrix = np.abs(original_distance_matrix - computed_distance_matrix)
    return computed_distance_matrix, difference_matrix

def create_ca_topology_from_coords(sequence, coords):
    """
    Creates a CA backbone topology from a protein sequence and 3D coordinates.

    Parameters:
    - sequence (str): Protein sequence as a string of amino acid single-letter codes.
    - coords (np.ndarray): 3D coordinates for each CA atom.

    Returns:
    - traj (md.Trajectory): MDTraj trajectory object containing the CA backbone topology and coordinates.
    """
    # Create an empty topology
    topology = md.Topology()

    # Add a chain to the topology
    chain = topology.add_chain()

    # Add residues and CA atoms to the topology
    for i, res in enumerate(sequence):
        # Add a residue
        residue = topology.add_residue(res, chain)

        # Add a CA atom to the residue
        ca_atom = topology.add_atom('CA', md.element.carbon, residue)

        # Connect the CA atom to the previous CA atom (if not the first residue)
        if i > 0:
            topology.add_bond(topology.atom(i - 1), ca_atom)

    # Ensure the coordinates are in the right shape (1, num_atoms, 3)
    coords = coords[np.newaxis, :, :]

    # Create an MDTraj trajectory object with the topology and coordinates
    traj = md.Trajectory(coords, topology)

    return traj

# Function to save the MDTraj trajectory to a specified file
def save_trajectory(traj, filename):
    """
    Saves the MDTraj trajectory to a specified file.

    Parameters:
    - traj (md.Trajectory): The MDTraj trajectory object to save.
    - filename (str): The name of the file to save the trajectory to.
    """
    traj.save(filename)

