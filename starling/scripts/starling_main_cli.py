from starling.frontend.ensemble_generation import generate
from argparse import ArgumentParser
from starling import configs

def main():
    # Initialize the argument parser
    parser = ArgumentParser(description="Generate distance maps using STARLING.")

    # Add command-line arguments corresponding to the parameters of the generate function
    parser.add_argument('user_input', type=str, help="Input sequences in various formats (file, string, list, or dict)")
    parser.add_argument('-c', '--conformations', type=int, default=configs.DEFAULT_NUMBER_CONFS, help=f"Number of conformations to generate (default: {configs.DEFAULT_NUMBER_CONFS})")
    parser.add_argument('-d', '--device', type=str, default=None, help="Device to use for predictions (default: None, auto-detected)")
    parser.add_argument('-s', '--steps', type=int, default=configs.DEFAULT_STEPS, help=f"Number of steps to run the DDPM model (default: {configs.DEFAULT_STEPS})")
    parser.add_argument('-m', '--method', type=str, default='mds', help="Method to use for generating 3D structures. Options are 'gd' or 'mds'. (default: 'mds')")
    parser.add_argument('-b', '--batch_size', type=int, default=configs.DEFAULT_BATCH_SIZE, help=f"Batch size to use for sampling (default: {configs.DEFAULT_BATCH_SIZE})")
    parser.add_argument('-o','--output_directory', type=str, default=".", help="Directory to save output (default: '.')")
    parser.add_argument('-r', '--return_structures', action='store_true', default=False, help="Return the 3D structures (default: False)")
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Enable verbose output (default: False)")
    parser.add_argument('--no-ddim', dest='ddim', default=True, action='store_false', help="Disable DDIM for sampling.")
    parser.add_argument('--disable_progress_bar', dest='progress_bar', action='store_false', default=True, help="Disable progress bar during generation (default: False)")

    # will need to update this default...
    parser.add_argument('--encoder',
                        type=str,
                        default=configs.DEFAULT_ENCODER_WEIGHTS_PATH,
                        help=f"Path to the encoder model (default: {configs.DEFAULT_ENCODER_WEIGHTS_PATH})")
    
    # will need to update this default...
    parser.add_argument('--ddpm',
                        type=str,
                        default=configs.DEFAULT_DDPM_WEIGHTS_PATH,
                        help=f"Path to the DDPM model (default: {configs.DEFAULT_DDPM_WEIGHTS_PATH}")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    
    # Call the generate function with parsed arguments
    generate(
        user_input=args.user_input,
        conformations=args.conformations,
        encoder=args.encoder,
        ddpm=args.ddpm,
        device=args.device,
        steps=args.steps,
        method=args.method,
        ddim=args.ddim,
        return_structures=args.return_structures,
        batch_size=args.batch_size,
        output_directory=args.output_directory,
        return_data=False,
        verbose=args.verbose,
        show_progress_bar=args.progress_bar,
    )

