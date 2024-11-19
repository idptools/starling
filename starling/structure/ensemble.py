import numpy as np
from starling import configs, utilities
from tqdm import tqdm
import torch


from soursop.sstrajectory import SSTrajectory

from starling.structure.coordinates import (
    create_ca_topology_from_coords,
    distance_matrix_to_3d_structure_gd,
    distance_matrix_to_3d_structure_mds,
)


class Ensemble:
    """
    Class to represent an ensemble of conformations of a protein chain.
    The ensemble is represented by a list of distance maps, where each 
    distance map is a 2D numpy array.

    """

    
    def __init__(self, distance_maps, sequence):
        """
        Initialize the ensemble with a list of distance maps and the sequence of the protein chain.

        Parameters
        ----------
        distance_maps : list of 2D numpy arrays
            List of distance maps, where each distance map is a 2D numpy array.
            Note this expects symmetrized distance maps.

        sequence : str
            Amino acid sequence of the protein chain.

        """

        ## TO DO: Add sanity checks for distance maps and sequence
        
        self.distance_maps = distance_maps
        self.sequence = sequence
        self.number_of_conformations = len(distance_maps)
        self.sequence_length = len(sequence)

        # initailize and then compute as needed
        self._rg_vals = []
        self._traj = None


    def rij(self, i, j, return_mean=False):
        """
        Compute the distance between residues i and j for each conformation 
        in the ensemble.

        Parameters
        ----------
        i : int
            Index of the first residue.

        j : int
            Index of the second residue.

        Returns
        -------
        list of float
            List of distances between residues i and j for each conformation 
            in the ensemble. If return_mean is set returns the mean value.

        
        """
        if i < 0 or i >= self.sequence_length:
            raise ValueError(f"Invalid residue index i: {i}")
        
        tmp =  []
        for d in self.distance_maps:
            tmp.append(d[i][j])

        if return_mean:
            return np.mean(tmp)
        else:
            return tmp
            

    def end_to_end_distance(self, return_mean=False):
        """
        Compute the end-to-end distance of the protein chain 
        for each conformation in the ensemble.

        Parameters
        ----------
        return_mean : bool
            If True, returns the mean end-to-end distance of the ensemble.

        Returns
        -------
        list of float
            List of end-to-end distances for each conformation in the ensemble.

        """
        
        tmp = self.rij(0, self.sequence_length-1)

        if return_mean:
            return np.mean(tmp)
        else:
            return tmp


    def radius_of_gyration(self, return_mean=False, force_recompute=False):
        """
        Compute the radius of gyration of the protein chain
        for each conformation in the ensemble.

        Parameters
        ----------
        return_mean : bool
            If True, returns the mean radius of gyration of the ensemble.
            Default is False.

        force_recompute : bool
            If True, forces recomputation of the radius of gyration, otherwise
            uses the cached value if previously computed.
            Default is False.

        
        Returns
        -------
        list of float
            List of radii of gyration for each conformation in the ensemble.
            If return_mean is set to true returns the mean value.

        """

        if len(self._rg_vals) == 0 or force_recompute = True:

            for d in self.distance_maps:
                distances = 0
                for i in range(self.sequence_length):
                    for j in range(self.sequence_length):
                        distances = distances + np.power(d[i][j],2)



                self._rg_vals.append(np.sqrt(distances/(2*np.power(self.sequence_length,2))))

        if return_mean:
            return np.mean(self._rg_vals)
        else:
            return self._rg_vals
                        

    def ensemble_trajectory(self,                               
                            method=configs.DEFAULT_STRUCTURE_GEN,
                            num_cpus_mds=configs.DEFAULT_CPU_COUNT_MDS,
                            num_mds_init=configs.DEFAULT_MDS_NUM_INIT,
                            device=None,
                            force_recompute=False):

        """
        Generate an ensemble of 3D structures from the distance maps using the 
        specified method.

        Parameters
        ----------
        method : str
            Method to use for generating the 3D structures. Must be one
            of 'mds' or 'gd'. Default is 'mds'.

        num_cpus_mds : int
            The number of CPUs to use for MDS. Default is 4

        num_mds_init : int
            Number of independent MDS jobs to execute. Default is 
            4. Note if goes up in principle more shots of finding
            a good solution but there is a performance hit unless
            num_cpus_mds == num_mds_init.

        device : str
            The device to use for predictions. Default is None. If None, the
            default device will be used. Default device is 'gpu'.
            This is MPS for apple silicon and CUDA for all other devices.
            If MPS and CUDA are not available, automatically falls back to CPU.

        """

        # define and sanitize the device
        device = utilities.check_device(device)

        # if no traj yet or we're focing to recompute...
        if self._traj is None or force_recompute:

            # as of 2024-11 mps does not support float64

            if method=='gd':
                coordinates = (
                    np.array(
                        [
                            distance_matrix_to_3d_structure_gd(
                                torch.from_numpy(dist_map),
                                num_iterations=10000,
                                learning_rate=0.05,
                                device=device,     
                                verbose=True,
                            )
                            for dist_map in self.distance_maps
                        ]
                    )
                    / configs.CONVERT_ANGSTROM_TO_NM
                )                 
            elif method=='mds':                
                coordinates = (
                    np.array(
                        [
                            distance_matrix_to_3d_structure_mds(
                                torch.from_numpy(dist_map),
                                n_jobs=num_cpus_mds,
                                n_init=num_mds_init,                                                           
                            )
                            for dist_map in self.distance_maps
                        ]
                    )
                    / configs.CONVERT_ANGSTROM_TO_NM
                )
            
            
            else:
                raise NotImplementedError("Method not implemented! We shouldn't have gotten this far.")

            # make traj and then use that to initailize a SOURSOP Trajectory object
            self._traj = SSTrajectory(TRJ=create_ca_topology_from_coords(self.sequence, coordinates)).proteinTrajectoryList[0]

        return self._traj


    
    def save_trajectory(self, filename_prefix, pdb_trajectory=False):
        """
        Save the ensemble trajectory to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the trajectory to, excluding
            file extensions which are added automatically.

        pdb_trajectory : bool
            If set to True, the output trajectory is ONLY saved as a PDB
            file. If set to false, it is saved as a single PDB structure 
            for topology and then the actual trajectory as an XTC file.

        """

        traj = self.ensemble_trajectory().traj

        if pdb_trajectory:
            traj.save_pdb(filename_prefix + ".pdb")
        else:
            traj[0].save_pdb(filename_prefix + ".pdb")
            traj.save_xtc(filename_prefix + ".xtc")
        
        
    def __len__(self):
        return len(self.distance_maps)

    def __str__(self):
        return f"Ensemble for chain of length {len(self.sequence)} with {len(self.distance_maps)} distance maps"

    def __repr__(self):
        return self.__str__()
        
        



