import argparse

import torch
from IPython import embed
from torch.utils.data import DataLoader

from starling.models.vae import (
    VAE,
    vae_loss_remove_padded,
    vae_loss_without_removing_padded,
)
from starling.training.myloader import MyDataset


# Testing the VAE
def test_vae(model, test_loader, device):
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data["input"]
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            if args.interpolate:
                loss = vae_loss_without_removing_padded(recon_batch, data, mu, logvar)
                test_loss += loss["loss"].item()
            else:
                # Here we want loss only over the non-padded region
                loss = vae_loss_remove_padded(recon_batch, data, mu, logvar)
                test_loss += loss["loss"].item()
            embed()

    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}".format(test_loss))


parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_file",
    type=str,
    default=None,
    help=("Path to the datafile in tsv format: <name> <data_path>"),
)

parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help=("Path to the trained model"),
)

parser.add_argument(
    "--latent_dim",
    type=int,
    default=128,
    help="The size of the latent 1D tensor",
)

parser.add_argument(
    "--kernel_size",
    type=int,
    default=5,
    help="The size of the kernel in Conv2d",
)

parser.add_argument(
    "--deep",
    type=int,
    default=5,
    help="How many convolutional layers (how deep should the network be)",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size for training",
)

parser.add_argument(
    "--normalize",
    type=str,
    default=None,
    help="""Whether to normalize the distance map. Options:[length, afrc, bond_length].
    Length is normalizing by np.sqrt(length), afrc by AFRC distance map, and bond_length 
    by the accumulated count of residues for each element within a distance map*bond_length""",
)

parser.add_argument(
    "--interpolate",
    action="store_true",
    help="Enable interpolation of distance maps through torch resizing using bicubic method",
)

parser.add_argument(
    "--gpu_id",
    type=int,
    default=0,
    help="GPU device ID for training",
)

args = parser.parse_args()

# Set up data loaders (assuming you have a dataset in a folder named 'data')
test_dataset = MyDataset(args.data_file, args)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

# Set up device
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

# Initialize VAE model
input_dim = 1  # Assuming distance map
latent_dim = args.latent_dim  # Adjust as needed

vae_model = VAE(input_dim, latent_dim, deep=args.deep, kernel_size=args.kernel_size).to(
    device
)

test_vae(vae_model, test_loader, device)
