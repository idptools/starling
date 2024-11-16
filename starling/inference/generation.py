
import os
import time

from tqdm import tqdm
import numpy as np
import torch

from starling.inference.model_loading import ModelManager
from starling.samplers.ddim_sampler import DDIMSampler
from starling.structure.coordinates import (
    compare_distance_matrices,
    create_ca_topology_from_coords,
    distance_matrix_to_3d_structure_gd,
    distance_matrix_to_3d_structure_mds,
)


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


def generate_backend(sequence_dict,
                     conformations,
                     device,
                     steps,
                     method,
                     ddim,
                     return_structures,
                     batch_size,
                     num_cpus_mds,
                     num_mds_init,
                     output_directory,
                     return_data,
                     verbose,
                     show_progress_bar,
                     model_manager=model_manager):

    """
    Backend function for generating the distance maps using STARLING.

    NOTE - this function does not sanity checking; to actually perform
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

    method : str
        The method to use for generating the 3D structure. 
    
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
        return None.

    verbose : bool
        Whether to print verbose output. Default is False.

    show_progress_bar : bool
        Whether to show a progress bar. Default is True.

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
        an np.ndarray of the distance maps as the value. 

        If return_structures=True, the dict will a key that is
        the sequence name + '_traj' and the value will be the 
        structure as a mdtraj.Trajectory object. 

        If output_directory is not none, the output will save to 
        the specified path.
    """

    # set CONVERT_ANGSTROM_TO_NM
    CONVERT_ANGSTROM_TO_NM = 10

    # get models. This will only load once even if we call this 
    # function multiple times. 
    encoder_model, diffusion  = model_manager.get_models(device=device)    

    # Construct a sampler
    if ddim:
        sampler = DDIMSampler(ddpm_model=diffusion, n_steps=steps)
    else:
        sampler = diffusion

    # get num_batchs and remaining samples
    num_batches = conformations // batch_size
    remaining_samples = conformations % batch_size

    # dictionary to hold distance maps and structures if applicable. 
    output_dict = {}

    # see if a progress bar is wanted. If it is, set it up. 
    if show_progress_bar:
        pbar = tqdm(total=len(sequence_dict))

    # iterate over sequence_dict
    for num, seq_name in enumerate(sequence_dict):

        start_time_prediction = time.time()
        # list to hold distance maps
        starling_dm=[]
    
        # get sequence
        sequence=sequence_dict[seq_name]
    
        # iterate over batches
        for batch in range(num_batches):
            distance_maps=sampler.sample(batch_size, labels=sequence)
            starling_dm.append(
                [
                    symmetrize_distance_map(dm[:, : len(sequence), : len(sequence)])
                    for dm in distance_maps
                ]
            )

        # iterate over remaining samples
        if remaining_samples > 0:
            distance_maps, *_ = sampler.sample(remaining_samples, labels=sequence)
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

        # if return_structures is True, generate 3D structure
        start_time_structure_generation = 0
        end_time_structure_generation = 0
        if return_structures:

            start_time_structure_generation = time.time()                
            if method=='gd':

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
            elif method=='mds':                
                coordinates = (
                    np.array(
                        [
                            distance_matrix_to_3d_structure_mds(
                                dist_map,
                                n_jobs=num_cpus_mds,
                                n_init=num_mds_init,
                            )
                            for dist_map in sym_distance_maps
                        ]
                    )
                    / CONVERT_ANGSTROM_TO_NM
                )
                
                
            else:
                raise NotImplementedError("Method not implemented! We shouldn't have gotten this far.")

            
            # make traj
            traj = create_ca_topology_from_coords(sequence, coordinates)
            end_time_structure_generation = time.time()

        # if we are saving things, save the things so we don't destroy memory usage
        if output_directory is not None:
            if verbose and num==0:
                print(f"Saving results to: {os.path.abspath(output_directory)}")
            if return_structures:
                # if we have structures, save the structures. 
                traj.save(os.path.join(output_directory, f"{seq_name}_STARLING.xtc"))

                # note save only first frame as a topology file
                traj[0].save(os.path.join(output_directory, f"{seq_name}_STARLING.pdb"))
            # save the distance maps

            # compress the distance maps to save space
            rounded_array = np.round(sym_distance_maps.detach().cpu().numpy(), decimals=2)
            rounded_array = rounded_array.astype(np.float32)
            np.save(os.path.join(output_directory, f"{seq_name}_STARLING_DM.npy"), rounded_array)
        else:
            # if not saving, we will add the info to the directory. 
            output_dict[seq_name] = sym_distance_maps.detach().cpu().numpy()
            if return_structures:
                output_dict[seq_name+'_traj'] = traj

        # update progress bar if we have one. 
        if show_progress_bar:
            pbar.update(1)

        if verbose:
            elapsed_time_structure_generation = end_time_structure_generation - start_time_structure_generation
            elapsed_time_prediction = end_time_prediction - start_time_prediction            
            total_time = elapsed_time_structure_generation + elapsed_time_prediction
            n_conformers = len(sym_distance_maps)
            print('----------------------------------------')
            print(f'Performance statisics for {seq_name}')
            print(f"Number of confomers                 : {n_conformers}")
            print(f"Total time for prediction           : {round(elapsed_time_prediction,2)}s ({round(100*(elapsed_time_prediction/total_time),2)}% of time)")
            print(f"Total time for structure generation : {round(elapsed_time_structure_generation,2)}s ({round(100*(elapsed_time_structure_generation/total_time),2)}% of time)")
            print(f"Time per conformer                  : {total_time/n_conformers}s")
            print('\n')

    # make sure we close the progress bar if we used one
    if show_progress_bar:
        pbar.close()

    


    # if we are not saving, return the output_dict
    if return_data:
        return output_dict



