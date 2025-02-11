from datetime import datetime

import numpy as np
import torch
from soursop.ssprotein import SSProtein
from soursop.sstrajectory import SSTrajectory
from tqdm import tqdm

from starling import configs, utilities
from starling._version import (
    __version__,
)
from starling.structure.coordinates import (
    create_ca_topology_from_coords,        
    generate_3d_coordinates_from_distances,
)


class Ensemble:
    """
    Class to represent an ensemble of conformations of a protein chain.
    The ensemble is represented by a 3D np.ndarray of N distance maps, where each
    distance map is a 2D numpy array.

    """

    def __init__(self, distance_maps, sequence, ssprot_ensemble=None):
        """
        Initialize the ensemble with a list of distance maps and the sequence
        of the protein chain.

        Parameters
        ----------
        distance_maps : np.ndarray
            3D Numpy array of shape (n_conformations, n_residues, n_residues).
            Note this this expects symmetrized distance maps.

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
            raise TypeError(
                "ssprot_ensemble must be a soursop.ssprotein.SSProtein object"
            )

        self.__metadata = {}
        self.__metadata["DEFAULT_ENCODER_WEIGHTS_PATH"] = (
            configs.DEFAULT_ENCODER_WEIGHTS_PATH
        )
        self.__metadata["DEFAULT_DDPM_WEIGHTS_PATH"] = configs.DEFAULT_DDPM_WEIGHTS_PATH
        self.__metadata["VERSION"] = __version__
        self.__metadata["DATE"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def __sanity_check_init(self, distance_maps, sequence, ssprot_ensemble):
        """
        Perform sanity checks on the distance maps and sequence.

        Parameters
        ----------
        distance_maps : np.ndarray
            3D Numpy array of shape (n_conformations, n_residues, n_residues).

        sequence : str
            Amino acid sequence of the protein chain.

        Raises
        ------
        ValueError
            If the distance maps are not a list of 2D numpy arrays or if the
            sequence is not a string.

        """

        if not isinstance(distance_maps, np.ndarray):
            raise ValueError("distance_maps must be a numpy ndarray")

        if not all([isinstance(d, np.ndarray) and d.ndim == 2 for d in distance_maps]):
            raise ValueError("distance_maps must be a list of 2D numpy arrays")

        if not all(
            [
                d.shape[0] == d.shape[1] and d.shape[0] == len(sequence)
                for d in distance_maps
            ]
        ):
            raise ValueError(
                "distance_maps must be square matrices with the same size as the sequence"
            )

        if not isinstance(sequence, str):
            raise ValueError("sequence must be a string")

        if not all(char in configs.VALID_AA for char in sequence):
            raise ValueError(
                "sequence must contain only valid amino acid characters ({configs.VALID_AA})"
            )

        if ssprot_ensemble is not None:
            if not isinstance(ssprot_ensemble, SSProtein):
                raise ValueError(
                    "ssprot_ensemble must be a soursop.ssprotein.SSProtein object"
                )

    def check_for_errors(
        self, remove_errors=False, verbose=True, rebuild_trajectory=False
    ):
        """
        Function which scans the ensemble and finds any frames which may be erroneous
        based on impossible intermolecular distances.

        Note if the ensemble has an SSProtein object associated with it and remove_errors
        is set to true, this will either rebuild the trajectory object from scratch (if
        rebuild_trajectory=True) or delete the SSProtein object (if rebuild_trajectory=False).

        Parameters
        ------------
        remove_errors : bool
            If set to True, the erroneous frames are removed from the ensemble.

        verbose : bool
            If set to True, the function will print out the indices of the erroneous
            frames.

        rebuild_trajectory : bool
            If True, and if remove_errors set to True, AND if the ensemble has trajectory
            (SSProtein) object associated with it, this will trigger the reconstruction
            of this trajectory object with the removed frames. If set to False, and if
            remove_errors set to True, AND if the ensemble has a trajectory (SSProtein)
            object associated with it, this will delete that object.

        Parameters
        ------------
        list
            List of indices of the erroneous frames (note if they have been removed)
            these indices no longer make sense...

        """

        bad_frames = []
        for idx, distance_map in enumerate(self.__distance_maps):
            if utilities.check_distance_map_for_error(
                distance_map, min_separation=1, max_separation=4
            ):
                bad_frames.append(idx)

        if len(bad_frames) > 0:
            if verbose:
                print(f"Found {len(bad_frames)} bad frames: {bad_frames}")

            # remove frames and update any derived values
            if remove_errors:
                if verbose:
                    print("Removing bad frames")
                self.__distance_maps = np.delete(
                    self.__distance_maps, bad_frames, axis=0
                )
                self._rg_vals = []
                self.number_of_conformations = len(self.__distance_maps)

                if self.__trajectory is not None:
                    if rebuild_trajectory:
                        # delete and zero
                        self.build_ensemble_trajectory(force_recompute=True)

        return bad_frames

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

        tmp = []
        for d in self.__distance_maps:
            tmp.append(d[i][j])

        tmp = np.array(tmp)

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

        tmp = self.rij(0, self.sequence_length - 1)

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
            return np.mean(self.__distance_maps, 0)
        else:
            return self.__distance_maps

    def contact_map(self, contact_thresh=11, return_mean=False, return_summed=False):
        """
        Return the collection of contact maps for the ensemble.
        Contacts are defined when residues are within a certain distance. Note
        only one of return_mean and return_summed can be set to True. If both
        are set to False, returns the array of instantaneous of contact maps.
        Else returns the averaged or the sum,ed of the contact maps.

        Parameters
        ----------
        contact_thresh : float
            Distance threshold for defining contacts. Default is 11
            Angstroms.

        return_mean : bool
            If True, the average contact map will be returned, meaning
            each element is between 0 and 1. Default is False.

        return_summed : bool
            If True, the summed contact map will be returned, meaning
            each element is an integer. Default is False.

        Returns
        -------
        np.array or list of np.array
            If return_mean is set to True, returns the mean distance map,
            otherwise returns the list of distance maps. Each distance map
            is a 2D numpy array.

        """
        # sanity check
        if return_mean and return_summed:
            raise ValueError("return_mean and return_summed cannot both be set to True")

        # get the distance maps
        dm = self.distance_maps(return_mean=False)

        if return_mean:
            return np.mean(np.array(dm < contact_thresh, dtype=int), 0)

        elif return_summed:
            return np.sum(np.array(dm < contact_thresh, dtype=int), 0)
        else:
            return np.array(dm < contact_thresh, dtype=int)

    def radius_of_gyration(
        self, return_mean=False, force_recompute=False, use_slow=False
    ):
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

    def local_radius_of_gyration(self, start, end, return_mean=False):
        """
        Return the local radius of gyration of the protein chain based on a subregion
        of the chain.

        Parameters
        ----------
        start : int
            The starting residue index.

        end : int
            The ending residue index.

        return_mean : bool
            If True, returns the mean radius of gyration of the ensemble.
            Default is False.

        Returns
        -------
        np.array or float
            Array of radii of gyration for each conformation in the ensemble.
            If return_mean is set to true returns the mean value as a float
        """

        local_rg = []
        for d in self.__distance_maps:
            distances = np.sum(np.square(d[start:end, start:end]))
            rg_val = np.sqrt(distances / (2 * np.power(end - start, 2)))
            local_rg.append(rg_val)

        local_rg = np.array(local_rg)

        if return_mean:
            return np.mean(local_rg)
        return local_rg

    def build_ensemble_trajectory(
        self,
        batch_size=100,
        num_cpus_mds=configs.DEFAULT_CPU_COUNT_MDS,
        num_mds_init=configs.DEFAULT_MDS_NUM_INIT,
        device=None,
        force_recompute=False,
        progress_bar=True,
    ):
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

        num_cpus_mds : int
            The number of CPUs to use for MDS. Default is 4 (set by
            configs.DEFAULT_CPU_COUNT_MDS)

        num_mds_init : int
            Number of independent MDS jobs to execute. NB: if this is
            increased this in principle means there are more chances
            of finding a good solution, but there is a performance hit
            unless num_cpus_mds >= num_mds_init. Default is
            4 (set by configs.DEFAULT_MDS_NUM_INIT).

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

        # define and sanitize the device (we cast to string to ensure its a string cos 
        # generate_3d_coordinates_from_distances expects a string
        device = str(utilities.check_device(device))
        
        # if no traj yet or we're focing to recompute...
        if self.__trajectory is None or force_recompute:

            # build the 3D coordinates
            coordinates = generate_3d_coordinates_from_distances(device, batch_size, num_cpus_mds, num_mds_init, self.__distance_maps, progress_bar=progress_bar)

            # make an mdtraj.Trajectory object and then use that to initailize a SOURSOP SSTrajectory object
            self.__trajectory = SSTrajectory(
                TRJ=create_ca_topology_from_coords(self.sequence, coordinates)
            ).proteinTrajectoryList[0]

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

    def save(
        self,
        filename_prefix,
        compress=False,
        reduce_precision=None,
        compression_algorithm="lzma",
        verbose=True,
    ):
        """
        Save the ensemble to a file in the STARLING format. Note this
        will add the .starling extension to the filename if not provided
        and will strip of any existing extension passed.

        Parameters
        ----------
        filename_prefix : str
            The name of the file to save the ensemble to, excluding
            file extensions which are added automatically. Note that if you
            provide a file extension it will be stripped off.

        compress : bool
            Whether to compress the file or not. Default is False.

        reduce_precision : bool
            Whether to reduce the precision of the distance map to a
            single decimal point and cast to float16 if possible.
            Default is None, and then sets to False if compression is
            False, but True if compression is True. However it can be
            manually over-ridden.

        compression_algorithm : str
            The compression algorithm to use. Options are 'gzip' and 'lzma'.
            `lzma` gives better compression if reduce_precision is set to True,
            but actually 'gzip' is better if reduce_precision is False. 'lzma'
            is also slower than 'gzip'. Default is 'lzma'.

        verbose : bool
            Flag to define how noisy we should be


        """
        utilities.write_starling_ensemble(
            self,
            filename_prefix,
            compress=compress,
            reduce_precision=reduce_precision,
            compression_algorithm=compression_algorithm,
            verbose=verbose,
        )

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
        """
        Return the number of conformations in the ensemble.
        """
        return len(self.__distance_maps)

    def __str__(self):
        """
        Return a string representation of the ensemble.
        """
        if self.__trajectory is not None:
            marker = "[X]"
        else:
            marker = "[ ]"
        return f"ENSEMBLE | len={len(self.sequence)}, ensemble_size={len(self)}, structures={marker}"

    def __repr__(self):
        """
        Return a string representation of the ensemble.
        """
        return self.__str__()


## ------------------------------------------ END OF CLASS DEFINITION


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

    # note there's exception handling in the utilities.py file, and we automatically
    # detect the compression algorithm based on the file extension
    return_dict = utilities.read_starling_ensemble(filename)

    # make sure we can extract out the core components
    try:
        sequence = return_dict["sequence"]
        distance_maps = return_dict["distance_maps"]
        traj = return_dict["traj"]
        DEFAULT_ENCODER_WEIGHTS_PATH = return_dict["DEFAULT_ENCODER_WEIGHTS_PATH"]
        DEFAULT_DDPM_WEIGHTS_PATH = return_dict["DEFAULT_DDPM_WEIGHTS_PATH"]
        VERSION = return_dict["VERSION"]
        DATE = return_dict["DATE"]
    except Exception as e:
        raise Exception(
            f"Error parsing STARLING ensemble data: {filename} [error 2]; error {e}"
        )

    try:
        E = Ensemble(distance_maps, sequence, traj)
    except Exception as e:
        raise Exception(
            f"Error initializing STARLING ensemble: {filename} [error 3]; error {e}"
        )

    # finally we over-write the metadata
    E._Ensemble__metadata["DEFAULT_ENCODER_WEIGHTS_PATH"] = DEFAULT_ENCODER_WEIGHTS_PATH
    E._Ensemble__metadata["DEFAULT_DDPM_WEIGHTS_PATH"] = DEFAULT_DDPM_WEIGHTS_PATH
    E._Ensemble__metadata["VERSION"] = VERSION
    E._Ensemble__metadata["DATE"] = DATE

    return E
