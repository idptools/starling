# starling.data

Despite its name, this directory mostly holds code rather than data: the tokenizer, the distributions and noise schedules used by the models, and the data loaders and helpers used for training.

## Manifest

* `tokenizer.py`: `StarlingTokenizer`, which maps amino-acid sequences to and from the integer tokens the models use (0 is the padding token).
* `distributions.py`: `DiagonalGaussianDistribution`, the VAE posterior over latents (`sample()` draws from it, `mode()` returns its mean, `kl()` gives the KL divergence).
* `schedulers.py`: linear, cosine and sigmoid beta (noise) schedules for the diffusion model.
* `positional_encodings.py`: fixed sinusoidal and learnable 1D/2D positional encodings.
* `VAE_loader_tar.py`: `VAEdataloader`, the PyTorch Lightning data module that streams distance maps from webdataset tar shards for VAE training (`starling-vae-train`).
* `ddpm_loader_tar.py`: `DDPMDataLoader`, the PyTorch Lightning data module that streams latents (or distance maps), tokenized sequences and ionic strengths from webdataset tar shards for diffusion training (`starling-ddpm-train`). If `train_size` or `val_size` is not set in the config, it is read from `metadata.json` in the dataset directory.
* `data_wrangler.py`: small helpers for one-hot encoding sequences, padding and symmetrizing distance maps, and reading HDF5 and TSV files.
* `argument_parser.py`: helpers that read a YAML configuration file into a dict (no defaults are merged in).
* `look_and_say.dat`: first entries of the "Look and Say" integer series, sequence [A005150](https://oeis.org/A005150). This is leftover sample data from the project template; nothing in STARLING uses it.

Both tar loaders expect a dataset directory containing `train/` and `validation/` subdirectories of `*.tar*` shards. Sample keys must end in the ionic strength, e.g. `<name>_150mM`, for the diffusion loader.
