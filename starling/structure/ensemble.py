import numpy as np
from datetime import datetime

from tqdm import tqdm
import torch

from soursop.sstrajectory import SSTrajectory
from soursop.ssprotein import SSProtein

from starling import __version__
from starling import configs, utilities

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

    
    def __init__(self, 
                 distance_maps, 
                 sequence, 
                 ssprot_ensemble=None):                 
        """
        Initialize the ensemble with a list of distance maps and the sequence of the protein chain.

        Parameters
        ----------
        distance_maps : np.ndarray
            3D Numpy array of shape (n_conformations, n_residues, n_residues). Note 
            this this expects symmetrized distance maps.

        sequence : str
            Amino acid sequence of the protein chain.

        ssprot_ensemble : soursop.ssprotein.SSProtein
            SOURSOP SSProtein object. If provided, the ensemble will be initialized
            using this.

        """

        # sanity check input
        self.__sanity_check_init(distance_maps, sequence, ssprot_ensemble)

        self.__distance_maps = distance_maps
        self.sequence = sequence
        self.number_of_conformations = len(distance_maps)
        self.sequence_length = len(sequence)        

        # initailize and then compute as needed
        self._rg_vals = []

        if ssprot_ensemble is None:
            self.__trajectory = None
        elif isinstance(ssprot_ensemble, SSProtein):
            self.__trajectory = ssprot_ensemble
        else:
            raise TypeError('ssprot_ensemble must be a soursop.ssprotein.SSProtein object')
        
        
        self.__metadata = {}
        self.__metadata['DEFAULT_ENCODER_WEIGHTS_PATH'] = configs.DEFAULT_ENCODER_WEIGHTS_PATH
        self.__metadata['DEFAULT_DDPM_WEIGHTS_PATH']    = configs.DEFAULT_DDPM_WEIGHTS_PATH
        self.__metadata['VERSION']    = __version__
        self.__metadata['DATE']       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  


    def __sanity_check_init(self, distance_maps, sequence, ssprot_ensemble):
        """
        Perform sanity checks on the distance maps and sequence.

        Parameters
        ----------
        distance_maps : list of 2D numpy arrays
            List of distance maps, where each distance map is a 2D numpy array.

        sequence : str
            Amino acid sequence of the protein chain.

        Raises
        ------
        ValueError
            If the distance maps are not a list of 2D numpy arrays or if the
            sequence is not a string.

        """

        if not isinstance(distance_maps, np.ndarray):
            raise ValueError("distance_maps must be a list of 2D numpy arrays")

        if not all([isinstance(d, np.ndarray) and d.ndim == 2 for d in distance_maps]):
            raise ValueError("distance_maps must be a list of 2D numpy arrays")
        
        if not all([d.shape[0] == d.shape[1] and d.shape[0] == len(sequence) for d in distance_maps]):
            raise ValueError("distance_maps must be square matrices with the same size as the sequence")
        
        if not isinstance(sequence, str):
            raise ValueError("sequence must be a string")
        
        VALID_AA = "ACDEFGHIKLMNPQRSTVWY"
        if not all(char in VALID_AA for char in sequence):
            raise ValueError("sequence must contain only valid amino acid characters ({VALID_AA})")
        
        if ssprot_ensemble is not None:
            if not isinstance(ssprot_ensemble, SSProtein):
                raise ValueError("ssprot_ensemble must be a soursop.ssprotein.SSProtein object")
            


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
        for d in self.__distance_maps:
            tmp.append(d[i][j])

        tmp  = np.array(tmp)

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
        
        
    def distance_maps(self, return_mean=False):
        """
        Return the collection of distance maps for the ensemble.

        Parameters
        ----------
        return_mean : bool
            If True, returns the mean distance map will be returned.
            Default is False.   

        Returns
        -------
        np.array or list of np.array
            If return_mean is set to True, returns the mean distance map,
            otherwise returns the list of distance maps. Each distance map
            is a 2D numpy array.

        """        
        if return_mean:
            return np.mean(self.__distance_maps,0)
        else:
            return self.__distance_maps    


    def radius_of_gyration(self, return_mean=False, force_recompute=False, use_slow=False):
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
        np.array or float
            Array of radii of gyration for each conformation in the ensemble.
            If return_mean is set to true returns the mean value as a float

        """

        if len(self._rg_vals) == 0 or force_recompute == True:
            for d in self.__distance_maps:
                distances = np.sum(np.square(d))
                rg_val = np.sqrt(distances / (2 * np.power(self.sequence_length, 2)))
                self._rg_vals.append(rg_val)

            self._rg_vals = np.array(self._rg_vals)

        if return_mean:
            return np.mean(self._rg_vals)
        else:
            return self._rg_vals
                        

    def build_ensemble_trajectory(self,                               
                            method=configs.DEFAULT_STRUCTURE_GEN,
                            num_cpus_mds=configs.DEFAULT_CPU_COUNT_MDS,
                            num_mds_init=configs.DEFAULT_MDS_NUM_INIT,
                            device=None,
                            force_recompute=False,
                            progress_bar=True):

        """
        Function that explicitly reconstructs a 3D ensemble of conformations
        using the distance maps. This happens automatically if the trajectory
        property is called, but this function allows for more control over the
        process. Specifically it allows you to specify the method used to generate
        the 3D structures, the number of CPUs to use, and the device to use for
        predictions. Note that if the 3D ensemble has already been reconstructed this
        function will NOT reconstructed the 3D ensemble unless force_recompute is set
        to True.

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

        force_recompute : bool
            If True, forces recomputation of the ensemble trajectory, otherwise
            uses the cached trajectory if previously computed.
            Default is False.

        progress_bar : bool
            If True, displays a progress bar when generating the ensemble 
            trajectory. Default is True.

        Returns
        -------
        soursop.sstrajectory.SSTrajectory
            The ensemble trajectory as a SOURSOP Trajectory object. Note that
            this object
            

        """

        # define and sanitize the device
        device = utilities.check_device(device)

        # if no traj yet or we're focing to recompute...
        if self.__trajectory is None or force_recompute:

            # check method before we initialize the progress bar
            if method not in ['mds', 'gd']:
                raise NotImplementedError("Method not implemented! We shouldn't have gotten this far.")

            # initialize progress bar
            if progress_bar == True:
                dm_generator = tqdm(self.__distance_maps)

            if method=='gd':

             
                # list comprehension version 
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
                            for dist_map in dm_generator
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
                            for dist_map in dm_generator
                        ]
                    )
                    / configs.CONVERT_ANGSTROM_TO_NM
                )
            
            
            else:
                raise Exception("Should not have gotten here. Method not implemented.")

            # make traj and then use that to initailize a SOURSOP Trajectory object
            self.__trajectory = SSTrajectory(TRJ=create_ca_topology_from_coords(self.sequence, coordinates)).proteinTrajectoryList[0]

        return self.__trajectory
    
    @property
    def trajectory(self):
        """
        Return the ensemble trajectory.

        Returns
        -------
        soursop.sstrajectory.SSTrajectory
            The ensemble trajectory as a SOURSOP Trajectory object.

        """
        if self.__trajectory is None:
            self.build_ensemble_trajectory()
        return self.__trajectory


    def save(self, filename_prefix):
        """
        Save the ensemble to a file in the STARLING format. Note this
        will add the .starling extension to the filename if not provided
        and will strip of any existing extension passed.

        Parameters
        ----------
        filename : str
            The name of the file to save the ensemble to, excluding
            file extensions which are added automatically.

        """
        utilities.write_starling_ensemble(self, filename_prefix)
    
    def save_trajectory(self, filename_prefix, pdb_trajectory=False):
        """
        Save the ensemble trajectory to a file. This ONLY saves the
        3D structural ensemble but does not save the STARLING-generated
        distance maps. We recommend using save() instead to save the 
        full STARLING object.

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

        traj = self.trajectory.traj

        if pdb_trajectory:
            traj.save_pdb(filename_prefix + ".pdb")
        else:
            traj[0].save_pdb(filename_prefix + ".pdb")
            traj.save_xtc(filename_prefix + ".xtc")


            
        
        
    def __len__(self):
        return len(self.__distance_maps)

    def __str__(self):
        if self.__trajectory is not None:
            marker = '[X]' 
        else:
            marker = '[ ]' 
        
        return f"ENSEMBLE | len={len(self.sequence)}, ensemble_size={len(self)}, structures={marker}"

    def __repr__(self):
        return self.__str__()
    
    
def load_ensemble(filename):
    """
    Function to read in a STARLING ensemble from a file and return the
    STARLING ensemble object.

    Parameters
    ---------------
    filename : str
        The filename to read the ensemble from (should be a .starling 
        file generated by STARLING)
    """
    
    # note there's exception handling in the utilities.py file
    return_dict =  utilities.read_starling_ensemble(filename)
    
    # make sure we can extract out the core components
    try:
        sequence = return_dict['sequence']
        distance_maps = return_dict['distance_maps']
        traj = return_dict['traj']
        DEFAULT_ENCODER_WEIGHTS_PATH = return_dict['DEFAULT_ENCODER_WEIGHTS_PATH']
        DEFAULT_DDPM_WEIGHTS_PATH = return_dict['DEFAULT_DDPM_WEIGHTS_PATH']
        VERSION = return_dict['VERSION']
        DATE = return_dict['DATE']
    except Exception as e:
        raise Exception(f"Error parsing STARLING ensemble data: {filename} [error 2]; error {e}")

    try:
        E = Ensemble(distance_maps, sequence, traj)
    except Exception as e:
        raise Exception(f"Error initializing STARLING ensemble: {filename} [error 3]; error {e}")
    
    # finally we over-write the metadata
    E._Ensemble__metadata['DEFAULT_ENCODER_WEIGHTS_PATH'] = DEFAULT_ENCODER_WEIGHTS_PATH
    E._Ensemble__metadata['DEFAULT_DDPM_WEIGHTS_PATH'] = DEFAULT_DDPM_WEIGHTS_PATH
    E._Ensemble__metadata['VERSION'] = VERSION
    E._Ensemble__metadata['DATE'] = DATE

    return E





