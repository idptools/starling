import sys
import os

from argparse import ArgumentParser

from starling. _version import __version__
from starling import configs
from starling.utilities import check_file_exists
from starling.frontend.ensemble_generation import generate, check_device

def print_starling():
    print('\n-------------------------------------------------------')
    print(f'     STARLING (version {__version__})           ')
    print('-------------------------------------------------------')
    

def print_info():

    # local imports if needed 
    print_starling()
    print('Using models at the following locations:')
    print(f'  VAE  model weights: {configs.DEFAULT_ENCODER_WEIGHTS_PATH}')
    print(f'  DDPM model weights: {configs.DEFAULT_DDPM_WEIGHTS_PATH}')
    print('-------------------------------------------------------')
    print('CONFIG INFO:')
    print( '  Default # of confs :', configs.DEFAULT_NUMBER_CONFS)
    print( '  Default batch size :', configs.DEFAULT_BATCH_SIZE)
    print( '  Default steps      :', configs.DEFAULT_STEPS)
    print( '  Default # of CPUs  :', configs.DEFAULT_CPU_COUNT_MDS)
    print( '  Default # MDS jobs :', configs.DEFAULT_MDS_NUM_INIT)
    print(f'  Default device     : {check_device(None)}')
    print('-------------------------------------------------------')
    print('Need help - please raise an issue on GitHub: https://github.com/idptools/starling/issues')
    
    print('\n')



def main():
    # Initialize the argument parser
    parser = ArgumentParser(description="Generate distance maps using STARLING.")

    # Add command-line arguments corresponding to the parameters of the generate function
    parser.add_argument('user_input', type=str, help="Input sequences in various formats (file, string, list, or dict)", nargs="?", default=None)
    parser.add_argument('-c', '--conformations', type=int, default=configs.DEFAULT_NUMBER_CONFS, help=f"Number of conformations to generate (default: {configs.DEFAULT_NUMBER_CONFS})")
    parser.add_argument('-d', '--device', type=str, default=None, help="Device to use for predictions (default: None, auto-detected)")
    parser.add_argument('-s', '--steps', type=int, default=configs.DEFAULT_STEPS, help=f"Number of steps to run the DDPM model (default: {configs.DEFAULT_STEPS})")
    parser.add_argument('-m', '--method', type=str, default='mds', help=f"Method to use for generating 3D structures. Options are 'gd' or 'mds'. (default: {configs.DEFAULT_STRUCTURE_GEN})")
    parser.add_argument('-b', '--batch_size', type=int, default=configs.DEFAULT_BATCH_SIZE, help=f"Batch size to use for sampling (default: {configs.DEFAULT_BATCH_SIZE})")
    parser.add_argument('-o','--output_directory', type=str, default=".", help="Directory to save output (default: '.')")
    parser.add_argument('--outname', type=str, default=None, help="If provided and a single sequence is provided, defines the prefix ahead of .pdb/.xtc/.npy extensions (default: None)")
    parser.add_argument('-r', '--return_structures', action='store_true', default=False, help="Return the 3D structures (default: False)")
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Enable verbose output (default: False)")
    parser.add_argument('--num-cpus', dest='num_cpus', type=int, default=configs.DEFAULT_CPU_COUNT_MDS, help=f"Sets the max number of CPUs to use. Default: {configs.DEFAULT_CPU_COUNT_MDS}.")
    parser.add_argument('--num-mds-init', dest='num_mds_init', type=int, default=configs.DEFAULT_MDS_NUM_INIT, help=f"Sets the number of MDS jobs to be run in parallel. More may give better reconstruction but requires 1:1 with #CPUs to avoid performance penalty. Default: {configs.DEFAULT_MDS_NUM_INIT}.")
    parser.add_argument('--no-ddim', dest='ddim', default=True, action='store_false', help="Disable DDIM for sampling.")
    parser.add_argument('--disable_progress_bar', dest='progress_bar', action='store_false', default=True, help="Disable progress bar during generation (default: False)")

    # will need to update this default...
    parser.add_argument('--info', action='store_true', default=False, help="Print STARLING information only")
    parser.add_argument('--version', action='store_true', default=False, help="Print STARLING version only")

    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # first. informational args overide anything else
    if args.info:
        print_info()
        sys.exit(0)
    elif args.version:
        print(__version__)
        sys.exit(0)        
    elif args.user_input is None:
        print(parser.format_usage())  # Print the command signature
        print("ERROR: STARLING requires user_input or --version", file=sys.stderr)        
        sys.exit(1)        
    else:        
        pass

    ### sanity checks
    # check if the output directory exists
    if not os.path.exists(args.output_directory):
        print(f"ERROR: Output directory {args.output_directory} does not exist.", file=sys.stderr)
        sys.exit(1)

    # check model files exist   
    if not check_file_exists(configs.DEFAULT_ENCODER_WEIGHTS_PATH):
        print(f"ERROR: VAE model weights file {configs.DEFAULT_ENCODER_WEIGHTS_PATH} does not exist.", file=sys.stderr)
        sys.exit(1)

    if not check_file_exists(configs.DEFAULT_DDPM_WEIGHTS_PATH):
        print(f"ERROR: DDPM model weights file {configs.DEFAULT_DDPM_WEIGHTS_PATH} does not exist.", file=sys.stderr)
        sys.exit(1)


    if args.verbose:
        print_starling()
    
    # Call the generate function with parsed arguments
    generate(
        user_input=args.user_input,
        conformations=args.conformations,
        device=args.device,
        steps=args.steps,
        method=args.method,
        ddim=args.ddim,
        return_structures=args.return_structures,
        batch_size=args.batch_size,
        num_cpus_mds=args.num_cpus,
        num_mds_init=args.num_mds_init,
        output_directory=args.output_directory,
        output_name=args.outname,
        return_data=False,
        verbose=args.verbose,
        show_progress_bar=args.progress_bar,
    )

