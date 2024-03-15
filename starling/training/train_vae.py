import argparse
from pathlib import Path

import torch
import torch.optim as optim
from IPython import embed
from torch.utils.data import DataLoader

from starling.models.vae import VAE, vae_loss
from starling.training.myloader import MyDataset


def train_vae(model, train_loader, validate_loader, optimizer, num_epochs, device):
    lowest_loss = float("inf")
    model.train()
    for epoch in range(num_epochs):
        accumulated_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data["input"]
            data = data.to(dtype=torch.float32)  # Is this a problem
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            accumulated_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0 and batch_idx != 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Average: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                        accumulated_loss / batch_idx,
                    )
                )
        validate_loss = 0
        for num, validate_data in enumerate(validate_loader):
            data = validate_data["input"]
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            validate_loss += vae_loss(recon_batch, data, mu, logvar).item()
            print(f"Validation Loss: {validate_loss/(num+1)}")

        if validate_loss / (num + 1) < lowest_loss:
            lowest_loss = validate_loss
            best_model_state = model.state_dict()
            Path(args.output_path).mkdir(exist_ok=True)
            torch.save(
                best_model_state,
                f"{args.output_path}/model_deep_{args.deep}_kernel_{args.kernel_size}_latent_{args.latent_dim}_norm_{args.normalize}.pt",
            )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--train_data",
    type=str,
    default=None,
    help="Path to the datafile in tsv format: <name> <data_path>",
)

parser.add_argument(
    "--validation_data",
    type=str,
    default=None,
    help="Path to the datafile in tsv format: <name> <data_path>",
)

parser.add_argument(
    "--output_path",
    type=str,
    default=".",
    help="Path to the directory to store the network in",
)

parser.add_argument(
    "--num_epochs",
    type=int,
    default=20,
    help="Number of epochs to do",
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
    default=16,
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
train_dataset = MyDataset(args.train_data, args)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

validate_dataset = MyDataset(args.validation_data, args)
validate_loader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True)

# Set up device
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

# Initialize VAE model
input_dim = 1  # Assuming distance map
latent_dim = args.latent_dim  # Adjust as needed

vae_model = VAE(input_dim, latent_dim, deep=args.deep, kernel_size=args.kernel_size).to(
    device
)

# Set up optimizer
optimizer = optim.Adam(vae_model.parameters(), lr=2e-3)
# optimizer = optim.SGD(vae_model.parameters(), lr=0.01, momentum=0.99, nesterov=True)

# Train the VAE
num_epochs = args.num_epochs
train_vae(vae_model, train_loader, validate_loader, optimizer, num_epochs, device)
