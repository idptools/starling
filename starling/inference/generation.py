from starling.models.diffusion import DiffusionModel
from starling.models.unet import UNetConditional
from starling.models.cvae import cVAE
from starling.structure.coordinates import compare_distance_matrices, loss_function, create_ca_topology_from_coords, save_trajectory, distance_matrix_to_3d_structure_gd
from IPython import embed
import torch
import mdtraj as md
import torch.optim as optim
import numpy as np
from scipy.spatial import distance_matrix
import time
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def plot_matrices(original, computed, difference, filename):
    """Plot the original, computed, and difference matrices using imshow and save to disk."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the original distance matrix
    im0 = axs[0].imshow(original, cmap='viridis')
    axs[0].set_title('Original Distance Matrix')
    axs[0].set_xlabel('Residue Index')
    axs[0].set_ylabel('Residue Index')
    fig.colorbar(im0, ax=axs[0])

    # Plot the computed distance matrix
    im1 = axs[1].imshow(computed, cmap='viridis')
    axs[1].set_title('Computed Distance Matrix (GD)')
    axs[1].set_xlabel('Residue Index')
    axs[1].set_ylabel('Residue Index')
    fig.colorbar(im1, ax=axs[1])

    # Plot the difference matrix
    im2 = axs[2].imshow(difference, cmap='viridis')
    axs[2].set_title('Difference Matrix (GD)')
    axs[2].set_xlabel('Residue Index')
    axs[2].set_ylabel('Residue Index')
    fig.colorbar(im2, ax=axs[2])

    # Save the plot to disk
    plt.savefig(filename)
    plt.close()

def symmetrize_distance_map(dist_map):
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

    return sym_dist_map


def compare_distance_matrices(original_distance_matrix, coords):
    computed_distance_matrix = distance_matrix(coords, coords)
    difference_matrix = np.abs(original_distance_matrix - computed_distance_matrix)
    return computed_distance_matrix, difference_matrix

def main():
    parser = ArgumentParser()
    parser.add_argument("--number_maps", type=int, default=1)
    #parser.add_argument("--sequence", type=str, default="A"*100)
    #parser.add_argument("--sequence", type=str, default="A"*384)
    parser.add_argument("--sequence", type=str, default="PKGS"*96)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=1000)

    args = parser.parse_args()

    UNet_model = UNetConditional(
        in_channels=1,
        out_channels=1,
        base=64,
        norm="group",
        blocks=[2, 2, 2],
        middle_blocks=2,
        labels_dim=384
    )

    encoder_model = cVAE.load_from_checkpoint(
        "/home/bnovak/github/starling/starling/models/trained_models/renamed_keys_model-kernel-epoch=09-epoch_val_loss=1.72.ckpt",
        map_location=args.device,
    )

    diffusion = DiffusionModel.load_from_checkpoint('/home/bnovak/projects/test_diffusion/diffusion_testing_my_UNet_attention/model-kernel-epoch=09-epoch_val_loss=0.05.ckpt', model=UNet_model, encoder_model=encoder_model, map_location=args.device)

    distance_maps, *_ = diffusion.sample(args.number_maps, labels = args.sequence, steps=args.steps)

    # Initialize an empty list to store symmetrized distance maps
    sym_distance_maps = []

    # Iterate over each distance map
    for dist_map in distance_maps:
        sym_dist_map = symmetrize_distance_map(dist_map)
        sym_distance_maps.append(sym_dist_map)

    # Convert the list back to a tensor
    sym_distance_maps = torch.stack(sym_distance_maps)
    coordinates = np.array([distance_matrix_to_3d_structure_gd(dist_map, num_iterations=10000, learning_rate=0.05, verbose=True)
                    for dist_map in sym_distance_maps
                ])

    computed_distance_matrix_gd, difference_matrix_gd = compare_distance_matrices(sym_distance_maps[0].cpu(), coordinates[0].squeeze())

    original_distance_matrix = sym_distance_maps[0].cpu()
    plot_matrices(original_distance_matrix, computed_distance_matrix_gd, difference_matrix_gd.cpu(), "distance_matrices.png")


    print(f"Min: {original_distance_matrix.min()}, Max: {original_distance_matrix.max()}")
    print(f"NaN values: {torch.isnan(original_distance_matrix).any()}")
    print(f"Inf values: {torch.isinf(original_distance_matrix).any()}")
    print(f"Symmetric: {torch.allclose(original_distance_matrix, original_distance_matrix.T)}")


    print("\nOriginal Distance Matrix")
    print(sym_distance_maps[0].cpu())
    
    print("\nComputed Distance Matrix (Gradient Descent):")
    print(computed_distance_matrix_gd)
    
    print("\nDifference Matrix (Gradient Descent):")
    print(difference_matrix_gd)


if __name__ == "__main__":
    main()
