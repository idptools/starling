from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS

from starling.models.diffusion import DiffusionModel
from starling.models.unet import UNetConditional
from starling.models.vae import VAE
from starling.samplers.ddim_sampler import DDIMSampler
from starling.structure.coordinates import (
    compare_distance_matrices,
    create_ca_topology_from_coords,
    distance_matrix_to_3d_structure_gd,
)


def distance_matrix_to_3d_structure(distance_matrix):
    # Initialize MDS with 3 components (for 3D)
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=42)

    # Fit the MDS model to the distance matrix
    coords = mds.fit_transform(distance_matrix.cpu())

    return coords


def plot_matrices(original, computed, difference, filename):
    """Plot the original, computed, and difference matrices using imshow and save to disk."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the original distance matrix
    im0 = axs[0].imshow(original, cmap="viridis")
    axs[0].set_title("Original Distance Matrix")
    axs[0].set_xlabel("Residue Index")
    axs[0].set_ylabel("Residue Index")
    fig.colorbar(im0, ax=axs[0])

    # Plot the computed distance matrix
    im1 = axs[1].imshow(computed, cmap="viridis")
    axs[1].set_title("Computed Distance Matrix (GD)")
    axs[1].set_xlabel("Residue Index")
    axs[1].set_ylabel("Residue Index")
    fig.colorbar(im1, ax=axs[1])

    # Plot the difference matrix
    im2 = axs[2].imshow(difference, cmap="viridis")
    axs[2].set_title("Difference Matrix (GD)")
    axs[2].set_xlabel("Residue Index")
    axs[2].set_ylabel("Residue Index")
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
    CONVERT_ANGSTROM_TO_NM = 10
    parser = ArgumentParser()
    parser.add_argument("--conformations", type=int, default=100)
    parser.add_argument("--input", type=str)
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--ddpm", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--method", type=str, default="mds")
    parser.add_argument("--ddim", action="store_true")
    parser.add_argument("--no_struct", action="store_true")
    parser.add_argument("--batch", type=int, default=100)

    args = parser.parse_args()

    UNet_model = UNetConditional(
        in_channels=1,
        out_channels=1,
        base=64,
        norm="group",
        blocks=[2, 2, 2],
        middle_blocks=2,
        labels_dim=512,
    )

    if args.encoder is not None:
        encoder_model = VAE.load_from_checkpoint(args.encoder, map_location=args.device)
    else:
        encoder_model = VAE.load_from_checkpoint(
            "/home/bnovak/github/starling/starling/models/trained_models/renamed_keys_model-kernel-epoch=09-epoch_val_loss=1.72.ckpt",
            map_location=args.device,
        )

    if args.ddpm is not None:
        diffusion = DiffusionModel.load_from_checkpoint(
            args.ddpm,
            model=UNet_model,
            encoder_model=encoder_model,
            map_location=args.device,
        )
    else:
        diffusion = DiffusionModel.load_from_checkpoint(
            "/home/bnovak/projects/test_diffusion/diffusion_testing_my_UNet_attention/model-kernel-epoch=09-epoch_val_loss=0.05.ckpt",
            model=UNet_model,
            encoder_model=encoder_model,
            map_location=args.device,
        )

    # Construct a sampler
    if args.ddim:
        sampler = DDIMSampler(ddpm_model=diffusion, n_steps=args.steps)
    else:
        sampler = diffusion

    with open(args.input, "r") as f:
        for line in f:
            filename, sequence = line.strip().split("\t")
                
            num_batches = args.conformations // args.batch
            remaining_samples = args.conformations % args.batch

            starling_dm = []

            for batch in range(num_batches):
                distance_maps, *_ = sampler.sample(
                    args.conformations, labels=sequence
                )
                starling_dm.append(
                    [
                        symmetrize_distance_map(dm[:, : len(sequence), : len(sequence)])
                        for dm in distance_map
                    ]
                )

            if remaining_samples > 0:
                distance_maps, *_ = sampler.sample(
                    remaining_samples, labels=sequence
                )
                starling_dm.append(
                    [
                        symmetrize_distance_map(dm[:, : len(sequence), : len(sequence)])
                        for dm in distance_map
                    ]
                )
            distance_maps = np.concatenate(starling_dm, axis=0)
            
            # Initialize an empty list to store symmetrized distance maps
            sym_distance_maps = []

            # Iterate over each distance map
            for dist_map in distance_maps:
                sym_dist_map = symmetrize_distance_map(
                    dist_map[:, : len(sequence), : len(sequence)]
                )
                sym_distance_maps.append(sym_dist_map)

            # Convert the list back to a tensor
            sym_distance_maps = torch.stack(sym_distance_maps)
            if not args.no_struct:
                if args.method == "gd":
                    coordinates = (
                        np.array(
                            [
                                distance_matrix_to_3d_structure_gd(
                                    dist_map,
                                    num_iterations=10000,
                                    learning_rate=0.05,
                                    verbose=True,
                                )
                                for dist_map in sym_distance_maps
                            ]
                        )
                        / CONVERT_ANGSTROM_TO_NM
                    )
                elif args.method == "mds":
                    # SCALE CORDINATES TO NM
                    coordinates = (
                        np.array(
                            [
                                distance_matrix_to_3d_structure(
                                    dist_map,
                                )
                                for dist_map in sym_distance_maps
                            ]
                        )
                        / CONVERT_ANGSTROM_TO_NM
                    )
                else:
                    raise NotImplementedError("Method not implemented")

                computed_distance_matrix_gd = []
                difference_matrix_gd = []

                for sym, coord in zip(sym_distance_maps, coordinates):
                    computed_dm, difference_dm = compare_distance_matrices(
                        sym.cpu(), coord.squeeze() * CONVERT_ANGSTROM_TO_NM
                    )
                    computed_distance_matrix_gd.append(computed_dm)
                    difference_matrix_gd.append(difference_dm)

                # computed_distance_matrix_gd, difference_matrix_gd = (
                #     compare_distance_matrices(
                #         sym_distance_maps[0].cpu(),
                #         coordinates[0].squeeze() * CONVERT_ANGSTROM_TO_NM,
                #     )
                # )

                # original_distance_matrix = sym_distance_maps[0].cpu()
                # plot_matrices(
                #     original_distance_matrix,
                #     computed_distance_matrix_gd,
                #     difference_matrix_gd.cpu(),
                #     "distance_matrices.png",
                # )

                # print(
                #     f"Min: {original_distance_matrix.min()}, Max: {original_distance_matrix.max()}"
                # )
                # print(f"NaN values: {torch.isnan(original_distance_matrix).any()}")
                # print(f"Inf values: {torch.isinf(original_distance_matrix).any()}")
                # print(
                #     f"Symmetric: {torch.allclose(original_distance_matrix, original_distance_matrix.T)}"
                # )

                print("\nOriginal Distance Matrix")
                print(sym_distance_maps[0].cpu())

                print("\nComputed Distance Matrix (Gradient Descent):")
                print(computed_distance_matrix_gd)

                print("\nDifference Matrix (Gradient Descent):")
                print(difference_matrix_gd)

                traj = create_ca_topology_from_coords(sequence, coordinates)
                traj.save(f"{filename}.xtc")
                traj.save(f"{filename}.pdb")
                np.save(f"{filename}_3D_struct_dm.npy", computed_distance_matrix_gd)
            np.save(
                f"{filename}_original_dm.npy", sym_distance_maps.detach().cpu().numpy()
            )


if __name__ == "__main__":
    main()
