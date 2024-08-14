import torch
import mdtraj as md
import torch.optim as optim
import numpy as np
from scipy.spatial import distance_matrix
import time




def compute_pairwise_distances(coords):
    """Function to compute the pairwise distances in 3D space

    Parameters
    ----------
    coords : np.ndarray
        An array of shape (n, 3) containing the 3D coordinates of n points

    Returns
    -------
    np.ndarray
        An array of pairwise distances between the points in 3D space
    """
    return torch.cdist(coords, coords)

def loss_function(original_distance_matrix, coords):
    """Function to compute the loss between the original distance matrix and the computed distance matrix

    Parameters
    ----------
    original_distance_matrix : np.ndarray
        An array of shape (n, n) containing the original pairwise distances between n points
    coords : np.ndarray
        An array of shape (n, 3) containing the 3D coordinates of n points

    Returns
    -------
    torch.Tensor
        The mean squared error between the original and computed distance matrices
    """
    computed_distances = compute_pairwise_distances(coords)
    return torch.nn.functional.mse_loss(computed_distances, original_distance_matrix)

def distance_matrix_to_3d_structure_gd(original_distance_matrix, num_iterations=5000, learning_rate=1e-3, verbose=True):
    """Function to reconstruct a 3D structure from a distance matrix using gradient descent

    Parameters
    ----------
    original_distance_matrix : _type_
        _description_
    num_iterations : int, optional
        _description_, by default 5000
    learning_rate : _type_, optional
        _description_, by default 1e-3
    verbose : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
    original_distance_matrix = torch.tensor(original_distance_matrix, dtype=torch.float32)

    coords = torch.randn((original_distance_matrix.size(0), 3), requires_grad=True)

    optimizer = optim.SGD([coords], lr=learning_rate, momentum=0.99, nesterov=True)

    for i in range(num_iterations):
        optimizer.zero_grad()

        loss = loss_function(original_distance_matrix, coords)

        loss.backward()

        optimizer.step()

        if i % 100 == 0 and verbose:
            print(f"Iteration {i}, Loss: {loss.item()}")

    return coords.detach().numpy()

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
