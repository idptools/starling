import gc
import os
import time
from datetime import datetime

import numpy as np
import torch
from soursop.sstrajectory import SSTrajectory
from tqdm.auto import tqdm

from starling import configs
from starling.inference.model_loading import ModelManager
from starling.samplers.ddim_sampler import DDIMSampler
from starling.structure.coordinates import (
    create_ca_topology_from_coords,
    generate_3d_coordinates_from_distances,
)
from starling.structure.ensemble import Ensemble

# initialize model_manager singleton. This happens when this module
# is imported to ensemble_generation, so we can use the
# same model_manager for all calls to generate_backend.
model_manager = ModelManager()


def symmetrize_distance_map(dist_map):
    """
    Symmetrize a distance map by replacing the lower triangle with the upper triangle values.

    Parameters
    ----------
    dist_map : torch.Tensor
        A 2D tensor representing the distance map.

    Returns
    -------
    torch.Tensor
        A symmetrized distance map.

    """

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

    return sym_dist_map.cpu()


def generate_backend(
    sequence_dict,
    conformations,
    device,
    steps,
    ddim,
    return_structures,
    batch_size,
    num_cpus_mds,
    num_mds_init,
    output_directory,
    return_data,
    verbose,
    show_progress_bar,
    show_per_step_progress_bar,
    pdb_trajectory,
    model_manager=model_manager,
):
    """
    Backend function for generating the distance maps using STARLING.

    NOTE - this function does VERY littel sanity checking; to actually perform
    predictions use starling.frontend.ensemble_generation. This is NOT
    a user facing function!

    Parameters
    ---------------
    sequence_dict : dict
        A dictionary with the sequence names as the key and the
       sequences as the values. These names will be used to write
       any output files (if writing is requested).

    ddpm : str
        The path to the DDPM model

    device : str
        The device to use for predictions.

    steps : int
        The number of steps to run the DDPM model.

    ddim : bool
        Whether to use DDIM for sampling.

    return_structures : bool
        Whether to return the 3D structure.

    batch_size : int
        The batch size to use for sampling.

    num_cpus_mds : int
        The number of CPUs to use for MDS. There
        is no point specifying more than the default
        number of MDS runs performed (defined in configs)

    output_directory : str or None
        If None, no output is saved.
        If not None, will save the output to the specified path.
        This includes the distance maps and if return_structures=True,
        the 3D structures.
        The distance maps are saved as .npy files with the names
        <sequence_name>_STARLING_DM.npy
        and the structures are save with the file names
        <sequence_name>_STARLING.xtc and <sequence_name>_STARLING.pdb.

    return_data : bool
        If True, will return the distance maps and structures (if generated)
        as a dictionary regardless of the output_directory. If False, will
        return None. Note the reason to set this to None is if you're
        predicting a large set of sequences this will save memory.

    verbose : bool
        Whether to print verbose output. Default is False.

    show_progress_bar : bool
        Whether to show a progress bar. Default is True.

    show_per_step_progress_bar : bool, optional
        whether to show progress bar per step.
    pdb_trajectory: bool
        Whether to save the trajectory as a PDB file. Default is False.

    model_manager : ModelManager
        A ModelManager object to manage loaded models.
        This lets us avoid loading the model iteratively
        when calling generate multiple times in a single
        session. Default is model_manager, which is initialized
        outside of this function code block. To update the path
        to the models, update the paths in config.py, which are
        read into the ModelManager object located the
        model_loading.py

    Returns
    ---------------
    dict or None:
        A dict with the sequence names as the key and
        a starling.ensembl.Ensemble objects for each
        sequence as values.

        If output_directory is not none, the output will save to
        the specified path.
    """

    overall_start_time = time.time()

    # get models. This will only load once even if we call this
    # function multiple times.
    encoder_model, diffusion = model_manager.get_models(device=device)

    # Construct a sampler
    if ddim:
        sampler = DDIMSampler(ddpm_model=diffusion, n_steps=steps)
    else:
        sampler = diffusion

    # get num_batchs and remaining samples
    num_batches = conformations // batch_size
    remaining_samples = conformations % batch_size

    if remaining_samples > 0:
        real_batch_count = num_batches + 1
    else:
        real_batch_count = num_batches

    # dictionary to hold distance maps and structures if applicable.
    output_dict = {}

    # see if a progress bar is wanted. If it is, set it up.
    # position here is 0, so it will be the first progress bar
    if show_progress_bar:
        pbar = tqdm(
            total=len(sequence_dict),
            position=0,
            desc="Progress through sequences",
            leave=True,
        )

    # iterate over sequence_dict
    for num, seq_name in enumerate(sequence_dict):
        ## -----------------------------------------
        ## Start of prediction cycle for this sequence

        start_time_prediction = time.time()

        # list to hold distance maps
        starling_dm = []

        # get sequence
        sequence = sequence_dict[seq_name]

        # iterate over batches for actual DDIM sampling
        for batch in range(num_batches):
            distance_maps = sampler.sample(
                batch_size,
                labels=sequence,
                show_per_step_progress_bar=show_per_step_progress_bar,
                batch_count=batch + 1,
                max_batch_count=real_batch_count,
            )
            starling_dm.append(
                [
                    symmetrize_distance_map(dm[:, : len(sequence), : len(sequence)])
                    for dm in distance_maps
                ]
            )

        # iterate over remaining samples
        if remaining_samples > 0:
            distance_maps = sampler.sample(
                remaining_samples,
                labels=sequence,
                show_per_step_progress_bar=show_per_step_progress_bar,
                batch_count=real_batch_count,
                max_batch_count=real_batch_count,
            )
            starling_dm.append(
                [
                    symmetrize_distance_map(dm[:, : len(sequence), : len(sequence)])
                    for dm in distance_maps
                ]
            )

        # concatenate symmetrized distance maps.
        sym_distance_maps = torch.cat(
            [torch.stack(batch) for batch in starling_dm], dim=0
        )

        end_time_prediction = time.time()

        # set time at which we start structure generation to 0
        start_time_structure_generation = time.time()

        # we initialize this to 0 and will update as needed (or not)
        end_time_structure_generation = time.time()

        # do ensemble reconstruction if requested
        if return_structures:
            coordinates = generate_3d_coordinates_from_distances(
                device,
                batch_size,
                num_cpus_mds,
                num_mds_init,
                sym_distance_maps,
                progress_bar=show_progress_bar,
            )

            # make traj as an sstrajectory object and extract out the ssprotein object
            ssprotein = SSTrajectory(
                TRJ=create_ca_topology_from_coords(sequence, coordinates)
            ).proteinTrajectoryList[0]

            end_time_structure_generation = time.time()

        # if no structures are requested, set ssprotein to None
        else:
            ssprotein = None

        # pull the distance maps out of the tensor and convert to numpy
        final_distance_maps = sym_distance_maps.detach().cpu().numpy()

        # create Ensemble object. Note if the ssprotein argument is None
        # this is expected and will initialize the ensemble without
        # structures
        E = Ensemble(final_distance_maps, sequence, ssprot_ensemble=ssprotein)

        # if we are saving things, save as we progress through so we generate
        # structures/DMs in situ
        if output_directory is not None:
            # num == 0 just means we are on the first sequence.
            if verbose and num == 0:
                print(f"Saving results to: {os.path.abspath(output_directory)}")

            # if we're saving structures do that first;
            if return_structures:
                # this saves both a topology (PDB) and a trajectory (XTC) file
                E.save_trajectory(
                    filename_prefix=os.path.join(
                        output_directory, seq_name + "_STARLING"
                    ),
                    pdb_trajectory=pdb_trajectory,
                )

            # save full ensemble
            E.save(os.path.join(output_directory, f"{seq_name}"))

        ## End of prediction cycle for this sequence
        ## -----------------------------------------

        # if we are returning data, add the data to the output_dict
        if return_data:
            output_dict[seq_name] = E

        # if not, force cleanup of things to save memory
        else:
            del E
            del final_distance_maps
            gc.collect()

        # update progress bar if we have one.
        if show_progress_bar:
            pbar.update(1)

        if verbose:
            elapsed_time_structure_generation = (
                end_time_structure_generation - start_time_structure_generation
            )
            elapsed_time_prediction = end_time_prediction - start_time_prediction
            total_time = elapsed_time_structure_generation + elapsed_time_prediction
            n_conformers = len(sym_distance_maps)

            print(
                f"\n\n##### SUMMARY OF SEQUENCE PREDICTION ({num + 1}/{len(sequence_dict)}) #####"
            )
            print(f"Sequence name                       : {seq_name}")
            print(f"Sequence length                     : {len(sequence)}")
            print(f"Number of conformers                : {n_conformers}")
            print(f"Number of steps                     : {steps}")
            print(
                f"Total time for prediction           : {round(elapsed_time_prediction, 2)}s ({round(100 * (elapsed_time_prediction / total_time), 2)}% of time)"
            )
            print(
                f"Total time for structure generation : {round(elapsed_time_structure_generation, 2)}s ({round(100 * (elapsed_time_structure_generation / total_time), 2)}% of time)"
            )
            print(f"Time per conformer                  : {total_time / n_conformers}s")
            print("\n")
        else:
            # changed from print to pass because if not verbose, we don't need to do anything.
            pass

    # make sure we close the progress bar if we used one
    if show_progress_bar:
        pbar.close()

    if verbose:
        # Convert total time to hours, minutes, and seconds
        overall_time = time.time() - overall_start_time
        total_hours = round(overall_time // 3600, 2)
        total_minutes = round((overall_time % 3600) // 60, 2)
        total_seconds = round(overall_time % 60, 2)
        print("-------------------------------------------------------")
        print(f"Summary of all predictions ({len(sequence_dict)} sequences)")
        print("-------------------------------------------------------")
        print(
            f"\nTotal time (all sequences, all I/O) : {total_hours} hrs {total_minutes} mins {total_seconds} secs"
        )

        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        print("STARLING predictions completed at:", formatted_datetime)
        print("")

    return output_dict
