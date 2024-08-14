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



def main():
    parser = ArgumentParser()
    parser.add_argument("--number_maps", type=int, default=2)
    parser.add_argument("--sequence", type=str, default="A"*100)
    parser.add_argument("--gpu", type=str, default="cuda:0")
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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder_model = cVAE.load_from_checkpoint(
        "/home/bnovak/github/starling/starling/models/trained_models/renamed_keys_model-kernel-epoch=09-epoch_val_loss=1.72.ckpt",
        map_location=device,
    )

    diffusion = DiffusionModel.load_from_checkpoint('/home/bnovak/projects/test_diffusion/diffusion_testing_my_UNet_attention/model-kernel-epoch=09-epoch_val_loss=0.05.ckpt', model=UNet_model, encoder_model=encoder_model, map_location=gpu)

    distance_maps, *_ = diffusion.sample(args.number_maps, labels = args.sequence, steps=args.steps)
    sym_distance_maps = [(dist_map + dist_map.T) /2 for dist_map in distance_maps]
    sym_distance_maps = torch.tensor([dist_map.fill_diagonal_(0) for dist_map in sym_distance_maps])
    embed()
    coordinates = [
            distance_matrix_to_3d_structure_gd(dist_map, num_iterations=5000, learning_rate=1e-3, verbose=True)
            for dist_map in sym_distance_maps
        ]
    
    print(coordinates)



if __name__ == "__main__":
    main()