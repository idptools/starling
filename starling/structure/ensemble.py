"""Ensemble representation and analysis helpers.

This module houses the :class:`Ensemble` container used throughout STARLING to
store distance-map ensembles, reconstruct 3D trajectories, and compute derived
biophysical observables. It also exposes convenience functions for error
diagnostics, Bayesian Maximum Entropy (BME) reweighting, and serialization to
the ``.starling`` format.

Key Features
------------
* Lightweight wrapper around distance maps with lazy computation of radius of
    gyration, hydrodynamic radius, contact maps, and more.
* Optional integration with :mod:`soursop` to build coordinate trajectories.
* Smooth interoperability with :class:`starling.structure.bme.BME` via cached
    reweighting results and helper diagnostics.

Example
-------
>>> from starling import generate
>>> ensemble = generate("GS" * 30, conformations=200, return_single_ensemble=True)
>>> mean_rg = ensemble.radius_of_gyration(return_mean=True)
>>> ensemble.save("example_ensemble")

"""

import warnings
from datetime import datetime
from typing import List, Optional, Tuple, Union

import numpy as np
from soursop.ssprotein import SSProtein
from soursop.sstrajectory import SSTrajectory

from starling import configs, utilities
from starling._version import __version__
from starling.structure.bme import BME, ThetaScanKwargs
from starling.structure.bme_utils import build_theta_grid
from starling.structure.coordinates import (
    create_ca_topology_from_coords,
    generate_3d_coordinates_from_distances,
)

# Absolute tolerance (Angstrom) used to check that every distance map is
# symmetric with a zero diagonal. Maps symmetrised by STARLING or computed from
# coordinates are exactly symmetric, so this only needs to absorb floating-point
# round-off; 0.001 A is far below any physically meaningful distance.
DISTANCE_MAP_SYMMETRY_ATOL_A = 1e-3


class Ensemble:
    """Distance-map backed representation of conformational ensembles.

    An :class:`Ensemble` stores pairwise residue distance maps for one or more
    sampled conformations and exposes convenience methods to analyse, reweight,
    and serialize the data.

    Parameters
    ----------
    distance_maps : np.ndarray
        Array of shape ``(n_conformations, n_residues, n_residues)`` containing
        symmetrised distance maps.
    sequence : str
        Amino-acid sequence corresponding to the ensemble.
    ssprot_ensemble : soursop.ssprotein.SSProtein, optional
        Existing SOURSOP trajectory to attach (used when coordinates already
        exist on disk).
    ionic_strength : float, optional
        Ionic strength in mM used to generate the ensemble, if known.

    Attributes
    ----------
    sequence : str
        Underlying amino-acid sequence.
    number_of_conformations : int
        Total number of conformations stored in the ensemble.
    sequence_length : int
        Number of residues in the sequence.

    Notes
    -----
    Most heavy operations (trajectory reconstruction, radius calculations,
    BME reweighting) are computed lazily and cached for reuse. Use
    :meth:`build_ensemble_trajectory` or :meth:`save_trajectory` to materialise
    3D coordinates on demand.
    """

    def __init__(
        self,
        distance_maps,
        sequence,
        ssprot_ensemble=None,
        ionic_strength: float | None = None,
    ):
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

        ionic_strength : float, optional
            Ionic strength in mM used to generate the ensemble, if known.

        """

        # sanity check input
        self.__sanity_check_init(distance_maps, sequence, ssprot_ensemble)

        self.__distance_maps = distance_maps
        self.sequence = sequence
        self.number_of_conformations = len(distance_maps)
        self.sequence_length = len(sequence)

        # initailize and then compute as needed
        self.__rg_vals = []
        self.__rh_vals = []
        self.__rh_settings_used = None

        # BME reweighting cache (the key records the inputs behind the result)
        self.__bme = None
        self.__bme_result = None
        self.__bme_cache_key: Optional[tuple] = None

        if ssprot_ensemble is None:
            self.__trajectory = None
        elif isinstance(ssprot_ensemble, SSProtein):
            self.__trajectory = ssprot_ensemble
        else:
            raise TypeError(
                "ssprot_ensemble must be a soursop.ssprotein.SSProtein object"
            )

        self.__metadata: dict[str, str | float] = {}
        self.__metadata["DEFAULT_ENCODER_WEIGHTS_PATH"] = "UNKNOWN (external ensemble)"
        self.__metadata["DEFAULT_DDPM_WEIGHTS_PATH"] = "UNKNOWN (external ensemble)"
        self.__metadata["VERSION"] = __version__
        self.__metadata["DATE"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if ionic_strength is not None:
            self.__metadata["IONIC_STRENGTH"] = float(ionic_strength)

    @property
    def ionic_strength(self) -> float | None:
        """Ionic strength in mM used to generate the ensemble, if known."""
        value = self.__metadata.get("IONIC_STRENGTH")
        return value if isinstance(value, float) else None

    @property
    def metadata(self) -> dict[str, str | float]:
        """Metadata recorded for this ensemble."""
        return self.__metadata

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
            If the distance maps are not a list of 2D numpy arrays, if any map
            is not symmetric with a zero diagonal (within
            DISTANCE_MAP_SYMMETRY_ATOL_A), or if the sequence is not a string.

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
                f"sequence must contain only valid amino acid characters ({configs.VALID_AA})"
            )

        # Every observable assumes full symmetric maps with a zero diagonal; an
        # upper-triangular map, for example, gives an Rg that is too small by
        # sqrt(2) and contact maps with an empty lower triangle. equal_nan=True
        # leaves non-finite maps to check_for_errors(), which flags them.
        asymmetric_maps = [
            idx
            for idx, d in enumerate(distance_maps)
            if not np.allclose(d, d.T, rtol=0, atol=DISTANCE_MAP_SYMMETRY_ATOL_A, equal_nan=True)
        ]
        if len(asymmetric_maps) > 0:
            raise ValueError(
                f"distance_maps must be symmetric, but {len(asymmetric_maps)} of {len(distance_maps)} "
                f"maps differ from their transpose by more than {DISTANCE_MAP_SYMMETRY_ATOL_A} A "
                f"(first offending map index: {asymmetric_maps[0]}). If your maps only hold one "
                "triangle, symmetrise them first, e.g. with starling.utilities.symmetrize_distance_maps()."
            )

        nonzero_diagonal_maps = [
            idx
            for idx, d in enumerate(distance_maps)
            if not np.allclose(np.diagonal(d), 0, rtol=0, atol=DISTANCE_MAP_SYMMETRY_ATOL_A)
        ]
        if len(nonzero_diagonal_maps) > 0:
            raise ValueError(
                f"distance_maps must have a zero diagonal, but {len(nonzero_diagonal_maps)} of "
                f"{len(distance_maps)} maps have a diagonal entry further than "
                f"{DISTANCE_MAP_SYMMETRY_ATOL_A} A from zero (first offending map index: "
                f"{nonzero_diagonal_maps[0]})."
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
        is set to True, the same frames are also removed from that trajectory so the
        structures stay in sync with the distance maps. By default this is done by
        slicing the existing trajectory, which keeps any relaxed or user-supplied
        coordinates. Only if rebuild_trajectory=True is the trajectory instead rebuilt
        from the remaining distance maps.

        Parameters
        ------------
        remove_errors : bool
            If set to True, the erroneous frames are removed from the ensemble.

        verbose : bool
            If set to True, the function will print out the indices of the erroneous
            frames.

        rebuild_trajectory : bool
            Only used if remove_errors is True AND the ensemble has a trajectory
            (SSProtein) object associated with it. If False (default), the erroneous
            frames are sliced out of the existing trajectory and the remaining
            coordinates are kept unchanged. If True, the existing coordinates are
            discarded and the trajectory is rebuilt from the remaining distance maps
            with build_ensemble_trajectory() (MDS/SMACOF reconstruction), so any
            relaxed or user-supplied structures are replaced.

        Returns
        ------------
        list
            List of indices of the erroneous frames (note if they have been removed)
            these indices no longer make sense...

        Raises
        ------
        ValueError
            If remove_errors is True, rebuild_trajectory is False and the attached
            trajectory does not have exactly one frame per distance map, so the
            frames to remove cannot be matched. Nothing is modified in this case.

        """

        bad_frames = []
        for idx, distance_map in enumerate(self.__distance_maps):
            if utilities.check_distance_map_for_error(
                distance_map, min_separation=1, max_separation=None
            ):
                bad_frames.append(idx)

        if len(bad_frames) > 0:
            if verbose:
                print(f"Found {len(bad_frames)} bad frames: {bad_frames}")

            # remove frames and update any derived values
            if remove_errors:
                slice_trajectory = self.__trajectory is not None and not rebuild_trajectory

                # trajectory frames can only be matched to distance maps one-to-one;
                # check before modifying anything so a mismatch leaves no partial state
                if slice_trajectory and self.__trajectory.n_frames != len(self.__distance_maps):
                    raise ValueError(
                        "Cannot remove bad frames from the trajectory: it has "
                        f"{self.__trajectory.n_frames} frames but there are "
                        f"{len(self.__distance_maps)} distance maps. Use "
                        "rebuild_trajectory=True to rebuild it from the distance maps instead."
                    )

                if verbose:
                    print("Removing bad frames")
                self.__distance_maps = np.delete(
                    self.__distance_maps, bad_frames, axis=0
                )
                self._invalidate_frame_derived_state()

                if self.__trajectory is not None:
                    if rebuild_trajectory:
                        # explicitly requested: replace the existing coordinates with
                        # a fresh reconstruction from the remaining distance maps
                        self.build_ensemble_trajectory(force_recompute=True)
                    else:
                        # slice the bad frames out so relaxed or user-supplied
                        # coordinates are kept (mirrors check_for_errors_trajectory)
                        bad_set = set(bad_frames)
                        good_frames = [
                            i for i in range(self.__trajectory.n_frames) if i not in bad_set
                        ]
                        self.__trajectory = SSTrajectory(
                            TRJ=self.__trajectory.traj.slice(good_frames)
                        ).proteinTrajectoryList[0]

        return bad_frames

    def check_for_errors_trajectory(self, remove_errors=False, verbose=True):
        """
        Scan the reconstructed 3D ensemble (the SSProtein trajectory) for
        frames with physically impossible inter-residue distances.

        This mirrors :meth:`check_for_errors`, but instead of inspecting the raw
        STARLING distance maps it inspects the per-frame CA-CA distance maps
        derived from the reconstructed 3D coordinates. It is therefore useful
        for catching reconstruction artefacts (e.g. a SMACOF embedding that
        introduces unphysical geometry even when the source distance map was
        well-behaved).

        Detection uses the same physical bound as :meth:`check_for_errors`
        (``utilities.check_distance_map_for_error``): a pair of residues
        separated by ``|i - j|`` positions can be at most ``|i - j|`` bond
        lengths apart.

        Note that the trajectory frames correspond one-to-one (and in order) to
        the distance maps. When ``remove_errors`` is True, the flagged frames are
        therefore removed from *both* the trajectory and the distance maps so the
        two representations stay in sync, and any cached derived values (Rg, Rh)
        are invalidated.

        Parameters
        ----------
        remove_errors : bool
            If True, the erroneous frames are removed from both the trajectory
            and the distance maps. Default is False.

        verbose : bool
            If True, print the number and indices of the erroneous frames (and a
            message when frames are removed). Default is True.

        Returns
        -------
        list
            List of indices of the erroneous frames (note that once they have
            been removed these indices no longer make sense).

        Raises
        ------
        RuntimeError
            If no SSProtein trajectory is associated with this ensemble. Build
            one first with :meth:`build_ensemble_trajectory`, or use
            :meth:`check_for_errors` to check the distance maps directly.

        """

        if self.__trajectory is None:
            raise RuntimeError(
                "No SSProtein trajectory is associated with this ensemble; "
                "build one with build_ensemble_trajectory() before calling "
                "check_for_errors_trajectory(), or use check_for_errors() to "
                "check the distance maps directly."
            )

        # per-frame CA-CA distance maps in Angstroms, shape (n_frames, N, N).
        # These are upper-triangular, which is fine for the error check (the
        # zeroed lower triangle never exceeds the positive distance bound).
        instantaneous_maps, _ = self.__trajectory.get_distance_map(
            return_instantaneous_maps=True, verbose=False
        )

        bad_frames = []
        for idx, distance_map in enumerate(instantaneous_maps):
            if utilities.check_distance_map_for_error(
                distance_map, min_separation=1, max_separation=None
            ):
                bad_frames.append(idx)

        if len(bad_frames) > 0:
            if verbose:
                print(f"Found {len(bad_frames)} bad frames: {bad_frames}")

            # remove frames and update any derived values
            if remove_errors:
                if verbose:
                    print("Removing bad frames")

                # slice the trajectory down to the good frames and rewrap as an
                # SSProtein (mirrors how build_ensemble_trajectory constructs it)
                bad_set = set(bad_frames)
                good_frames = [
                    i for i in range(len(instantaneous_maps)) if i not in bad_set
                ]
                self.__trajectory = SSTrajectory(
                    TRJ=self.__trajectory.traj.slice(good_frames)
                ).proteinTrajectoryList[0]

                # keep the distance maps (and derived values) in sync
                self.__distance_maps = np.delete(
                    self.__distance_maps, bad_frames, axis=0
                )
                self._invalidate_frame_derived_state()

        return bad_frames

    def _invalidate_frame_derived_state(self) -> None:
        self.number_of_conformations = len(self.__distance_maps)
        self.__rg_vals = []
        self.__rh_vals = []
        self.__bme = None
        self.__bme_result = None
        self.__bme_cache_key = None

    def rij(self, i, j, return_mean=False, use_bme_weights=False):
        """
        Compute the distance between residues i and j for each conformation
        in the ensemble.

        Parameters
        ----------
        i : int
            Index of the first residue.

        j : int
            Index of the second residue.

        return_mean : bool
            If True, returns the mean distance. Default is False.

        use_bme_weights : bool
            If True, use BME-optimized weights for computing mean.
            Only applicable if return_mean=True. Default is False.

        Returns
        -------
        float or np.ndarray
            If return_mean=True, returns weighted mean distance.
            Otherwise, returns array of distances for each conformation.

        """
        if i < 0 or i >= self.sequence_length:
            raise ValueError(f"Invalid residue index i: {i}")

        if j < 0 or j >= self.sequence_length:
            raise ValueError(f"Invalid residue index j: {j}")

        tmp = self.__distance_maps[:, i, j]

        if return_mean:
            if use_bme_weights:
                weights = self._get_weights(use_bme_weights=True)
                return np.sum(tmp * weights)
            else:
                return np.mean(tmp)
        else:
            return tmp

    def end_to_end_distance(self, return_mean=False, use_bme_weights=False):
        """
        Compute the end-to-end distance of the protein chain
        for each conformation in the ensemble.

        Parameters
        ----------
        return_mean : bool
            If True, returns the mean end-to-end distance of the ensemble.

        use_bme_weights : bool
            If True, use BME-optimized weights for computing mean.
            Only applicable if return_mean=True. Default is False.

        Returns
        -------
        float or np.ndarray
            If return_mean=True, returns weighted mean distance.
            Otherwise, returns array of end-to-end distances for each conformation.

        """

        tmp = self.rij(0, self.sequence_length - 1)

        if return_mean:
            if use_bme_weights:
                weights = self._get_weights(use_bme_weights=True)
                return np.sum(tmp * weights)
            else:
                return np.mean(tmp)
        else:
            return tmp

    def distance_maps(self, return_mean=False, use_bme_weights=False):
        """
        Return the collection of distance maps for the ensemble.

        Parameters
        ----------
        return_mean : bool
            If True, returns the mean distance map will be returned.
            Default is False.
        use_bme_weights : bool
            If True, use BME-optimized weights for computing mean.
            Only applicable if return_mean=True. Default is False.
        Returns
        -------
        np.array or list of np.array
            If return_mean is set to True, returns the mean distance map,
            otherwise returns the list of distance maps. Each distance map
            is a 2D numpy array.

        """
        if return_mean:
            if use_bme_weights:
                weights = self._get_weights(use_bme_weights=True)
                # Reshape weights for broadcasting: (n_frames, 1, 1)
                weights_reshaped = weights[:, np.newaxis, np.newaxis]
                # Weighted average: sum(weight_i * distance_map_i)
                return np.sum(self.__distance_maps * weights_reshaped, axis=0)
            else:
                return np.mean(self.__distance_maps, axis=0)
        else:
            return self.__distance_maps

    # TODO: BME weights?
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
        np.array
            If return_mean is set to True, returns the fraction of
            conformations in which each residue pair is in contact, shape
            (n_residues, n_residues), values between 0 and 1. If return_summed
            is set to True, returns the number of conformations in which each
            pair is in contact (same shape, integers). Otherwise returns the
            instantaneous contact maps, shape (n_conformations, n_residues,
            n_residues), with 1 for a contact and 0 otherwise.

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
        self, return_mean=False, force_recompute=False, use_bme_weights=False
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

        use_bme_weights : bool
            If True, use BME-optimized weights for computing mean.
            Only applicable if return_mean=True. Default is False.

        Returns
        -------
        np.array or float
            Array of radii of gyration for each conformation in the ensemble.
            If return_mean is set to true returns the mean value as a float.
            The array is a copy, so editing it does not change the cache.

        """
        if len(self.__rg_vals) == 0 or force_recompute:
            self.__rg_vals = []
            for d in self.__distance_maps:
                distances = np.sum(np.square(d))
                rg_val = np.sqrt(distances / (2 * np.power(self.sequence_length, 2)))
                self.__rg_vals.append(rg_val)

            self.__rg_vals = np.array(self.__rg_vals)

        if return_mean:
            if use_bme_weights:
                weights = self._get_weights(use_bme_weights=True)
                return np.sum(self.__rg_vals * weights)
            else:
                return np.mean(self.__rg_vals)
        else:
            # return a copy so in-place edits by the caller cannot corrupt the cache
            return self.__rg_vals.copy()

    def local_radius_of_gyration(
        self, start, end, return_mean=False, use_bme_weights=False
    ):
        """
        Return the local radius of gyration of the protein chain based on a subregion
        of the chain. The region follows Python slice conventions: residues
        start, start+1, ..., end-1 (0-indexed, end exclusive).

        Parameters
        ----------
        start : int
            The starting residue index (0-indexed, inclusive).

        end : int
            The ending residue index (0-indexed, exclusive). Must satisfy
            0 <= start < end <= sequence_length.

        return_mean : bool
            If True, returns the mean radius of gyration of the ensemble.
            Default is False.

        use_bme_weights : bool
            If True, use BME-optimized weights for computing mean.
            Only applicable if return_mean=True. Default is False.

        Returns
        -------
        np.array or float
            Array of radii of gyration for each conformation in the ensemble.
            If return_mean is set to true returns the mean value as a float

        Raises
        ------
        ValueError
            If the region is not 0 <= start < end <= sequence_length; an end
            beyond the chain would otherwise be silently truncated by the slice
            while still counting in the (end - start) normalisation.
        """

        if not (0 <= start < end <= self.sequence_length):
            raise ValueError(
                f"Invalid region start={start}, end={end}: need 0 <= start < end <= "
                f"{self.sequence_length} (end is exclusive)"
            )

        local_rg = []
        for d in self.__distance_maps:
            distances = np.sum(np.square(d[start:end, start:end]))
            rg_val = np.sqrt(distances / (2 * np.power(end - start, 2)))
            local_rg.append(rg_val)

        local_rg = np.array(local_rg)

        if return_mean:
            if use_bme_weights:
                weights = self._get_weights(use_bme_weights=True)
                return np.sum(local_rg * weights)
            else:
                return np.mean(local_rg)
        return local_rg

    # TODO BME weights?
    def hydrodynamic_radius(
        self,
        return_mean=False,
        force_recompute=False,
        mode="nygaard",
        alpha1=0.216,
        alpha2=4.06,
        alpha3=0.821,
    ):
        """
        Compute the hydrodynamic radius of each conformation using either the
        Kirkwood-Riseman (mode='kr') or Nygaard (mode='nygaard') equation (default)
        is Nygaard.

        The Kirkwood-Riseman [1] equation may be more accurate when computing
        the Rh for comparison with NMR-derived Rh values, as reported by Pesce
        et al. [2]. The Nygaard equation is a more general form of the
        Kirkwood-Riseman equation, and may offer better agreement with
        values obtained from dynamic light scattering (DLS) experiments [3].

        For 'kr' (Kirkwood-Riseman mode), the alpha1/2/3 arguments are
        ignored, as these are only used with the Nygaard mode. Each
        conformation's Rh is computed from the Kirkwood-Riseman expression [1]

            1/Rh = (1/N^2) * sum_{i != j} 1/r_ij

        over all N residues, i.e. Rh = N^2 / (2 * sum_{i<j} 1/r_ij), which
        matches SOURSOP >= 2.0.6.

        For 'nygaard' mode, the arguments (alpha1/2/3) are used, and should
        not be altered to recapitulate behaviour defined by Nygaard et al.
        Default values here are alpha1=0.216, alpha2=4.06 and alpha3=0.821.
        The Nygaard equation is parameterised for chains much longer than a
        few residues; for very short chains its Rg/Rh ratio can be zero or
        negative, in which case a ValueError is raised and mode='kr' should
        be used instead.

        NB: If an Rh value is computed and then re-requested with the same
        mode (and, for 'nygaard', the same alpha1/2/3), the cached value is
        returned. This is to avoid recomputing the Rh value if it has already
        been computed. If you want to recompute the Rh value, set
        force_recompute=True. Also, if Rh with a different mode or different
        alpha values is requested, the value is recomputed automatically. This
        is to avoid a situation where you request Rh for one setting but
        actually get a (cached) value for another.

        [1]  Kirkwood & Riseman,(1948). The Intrinsic Viscosities
        and Diffusion Constants of Flexible Macromolecules in Solution.
        The Journal of Chemical Physics, 16(6), 565-573 (J. Chem. Phys. 16, 565).

        [2] Pesce et al. Assessment of models for calculating the hydrodynamic
        radius of intrinsically disordered proteins. Biophys. J. 122,
        310-321 (2023).

        [3] Nygaard et al.  An Efficient Method for Estimating the Hydrodynamic
        Radius of Disordered Protein Conformations. Biophys J. 2017;113: 550-557.

        Parameters
        ----------
        return_mean : bool
            If True, returns the mean hydrodynamic radius of the
            ensemble. Default is False.

        force_recompute : bool
            If True, forces recomputation of the hydrodynamic radius,
            otherwise uses the cached value if previously computed.
            Default is False.

        mode : str
            The mode to use for computing the hydrodynamic radius.
            Options are 'kr' (Kirkwood-Riseman) or 'nygaard' (Nygaard).
            Default is 'nygaard'.

        alpha1 : float
           First parameter in equation (7) from Nygaard et al. Default = 0.216

        alpha2 : float
           Second parameter in equation (7) from Nygaard et al. Default = 4.06

        alpha3 : float
           Third parameter in equation (7) from Nygaard et al. Default = 0.821

        Returns
        -------
        np.array or float
            Array of hydrodynamic radii (Angstroms) for each conformation in
            the ensemble. If return_mean is set to true returns the mean value
            as a float. The array is a copy, so editing it does not change
            the cache.

        Raises
        ------
        ValueError
            If mode is not 'kr' or 'nygaard', or if in 'nygaard' mode the
            Rg/Rh ratio is not positive for one or more conformations.
        """

        # check the mode
        mode = mode.lower()
        if mode not in ["kr", "nygaard"]:
            raise ValueError("mode must be either 'kr' or 'nygaard'")

        # cache key: Nygaard values depend on the alphas, Kirkwood-Riseman ones do not
        if mode == "nygaard":
            rh_settings = (mode, float(alpha1), float(alpha2), float(alpha3))
        else:
            rh_settings = (mode,)

        # if we want to recompute...
        if len(self.__rh_vals) == 0 or force_recompute or rh_settings != self.__rh_settings_used:
            # Nygaard mode
            if mode == "nygaard":
                # first compute the rg
                rg = self.radius_of_gyration()

                # precompute
                N_033 = np.power(len(self.sequence), 0.33)
                N_060 = np.power(len(self.sequence), 0.60)

                Rg_over_Rh = (
                    (alpha1 * (rg - alpha2 * N_033)) / (N_060 - N_033)
                ) + alpha3

                # Eq. 7 of Nygaard et al. is fitted to longer chains; for very
                # short ones Rg/Rh can be <= 0 (always for N=3), which would give
                # a negative or divergent Rh. "not > 0" also catches NaN.
                n_invalid = int(np.sum(~(Rg_over_Rh > 0)))
                if n_invalid > 0:
                    raise ValueError(
                        f"The Nygaard Rg/Rh ratio is not positive for {n_invalid} of "
                        f"{len(Rg_over_Rh)} conformations (chain length {len(self.sequence)}), "
                        "so the Nygaard Rh would be negative or divergent. The Nygaard equation "
                        "is parameterised for longer chains; use mode='kr' instead."
                    )

                Rh = (1 / Rg_over_Rh) * rg

                # assign the values to the class
                self.__rh_vals = Rh
                self.__rh_settings_used = rh_settings

            # Kirkwood-Riseman mode
            elif mode == "kr":
                # Kirkwood & Riseman (1948): 1/Rh = (1/N^2) sum_{i != j} 1/r_ij per
                # conformation. Summing each unordered pair once gives
                # Rh = N^2 / (2 sum_{i<j} 1/r_ij), as in SOURSOP >= 2.0.6
                n_residues = len(self.sequence)
                upper_i, upper_j = np.triu_indices(n_residues, k=1)

                rh_per_frame = []
                for d in self.__distance_maps:
                    pair_distances = d[upper_i, upper_j].astype(np.float64)
                    inverse_distance_sum = np.sum(1.0 / pair_distances)
                    rh_per_frame.append(n_residues**2 / (2.0 * inverse_distance_sum))

                Rh = np.array(rh_per_frame)

                # assign the values to the class
                self.__rh_vals = Rh
                self.__rh_settings_used = rh_settings

            # read from precomputed values
            else:
                raise Exception("Invalid mode specified. Must be 'kr' or 'nygaard'")

        # else use the precomputed values
        else:
            Rh = self.__rh_vals

        if return_mean:
            return np.mean(Rh)
        else:
            # return a copy so in-place edits by the caller cannot corrupt the cache
            return Rh.copy()

    def build_ensemble_trajectory(
        self,
        batch_size=100,
        device=None,
        force_recompute=False,
        progress_bar=True,
    ):
        """
        Function that explicitly reconstructs a 3D ensemble of conformations
        using the distance maps. This happens automatically if the trajectory
        property is called, but this function allows for more control over the
        process. Specifically it allows you to specify the device to use for
        generating the 3D structures. Note that if the 3D ensemble has already
        been reconstructed this function will NOT reconstruct the 3D ensemble
        unless force_recompute is set to True.

        Parameters
        ----------
        batch_size : int
            Number of distance maps reconstructed together in each batch on
            the device. Larger batches are faster but use more memory.
            Default is 100.

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
        soursop.ssprotein.SSProtein
            The ensemble trajectory as a SOURSOP SSProtein object (one frame
            per distance map; the underlying mdtraj coordinates are in nm).

        """

        # define and sanitize the device (we cast to string to ensure its a string cos
        # generate_3d_coordinates_from_distances expects a string
        device = str(utilities.check_device(device))

        # if no traj yet or we're focing to recompute...
        if self.__trajectory is None or force_recompute:
            # build the 3D coordinates
            coordinates = generate_3d_coordinates_from_distances(
                device,
                batch_size,
                self.__distance_maps,
                progress_bar=progress_bar,
            )

            # make an mdtraj.Trajectory object and then use that to initailize a SOURSOP SSTrajectory object
            self.__trajectory = SSTrajectory(
                TRJ=create_ca_topology_from_coords(self.sequence, coordinates)
            ).proteinTrajectoryList[0]

        return self.__trajectory

    @property
    def has_structures(self):
        """
        Check if the ensemble has 3D structures.

        Returns
        -------
        bool
            True if the ensemble has 3D structures, False otherwise.

        """
        if self.__trajectory is None:
            return False
        return True

    @property
    def trajectory(self):
        """
        Return the ensemble trajectory, reconstructing it from the distance
        maps first if no trajectory exists yet.

        Returns
        -------
        soursop.ssprotein.SSProtein
            The ensemble trajectory as a SOURSOP SSProtein object.

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
            Whether to round the distance map to a single decimal point
            (0.1 Å) and cast it to float16 if possible. float16 resolves
            0.1 Å only below 128 Å; above that the spacing between stored
            values is 0.125 Å (128-256 Å) or 0.25 Å (256-512 Å) (see
            starling.utilities.write_starling_ensemble).
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
        filename_prefix : str
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
        if self.has_structures:
            marker = "[X]"
        else:
            marker = "[ ]"
        return f"ENSEMBLE | len={len(self.sequence)}, ensemble_size={len(self)}, structures={marker}"

    def __repr__(self):
        """
        Return a string representation of the ensemble.
        """
        return self.__str__()

    def reweight_bme(
        self,
        observables,
        calculated_values,
        theta=None,
        theta_range=(0.01, 10.0),
        theta_n_points=15,
        max_iterations=50000,
        optimizer="L-BFGS-B",
        initial_weights=None,
        force_recompute=False,
        verbose=True,
        show=False,
        save_theta_scan_plot=None,
        theta_method: str = "perpendicular",  # new: choose knee method ("perpendicular" or "curvature")
    ):
        """
        Perform Bayesian Maximum Entropy (BME) reweighting of the ensemble
        to better match experimental observables while minimizing bias.

        This method creates optimized weights for each frame that balance
        fitting experimental data with maintaining ensemble diversity: the
        weights minimise (1/2) * chi^2 + theta * D_KL(weights || initial
        weights). A successful result is cached and can be used with other
        ensemble methods by setting use_bme_weights=True. The cached result is
        only returned by a later call whose inputs (observables,
        calculated_values, theta settings, optimizer settings and
        initial_weights) are identical; any change triggers a new fit.

        Parameters
        ----------
        observables : list[ExperimentalObservable]
            List of experimental observables to fit. Each observable should
            be an ExperimentalObservable object with value, uncertainty, and
            constraint type.

        calculated_values : np.ndarray
            Array of calculated observable values for each frame.
            Shape: (n_frames, n_observables)

        theta : float or None, optional
            Regularization parameter controlling the trade-off between fitting
            experimental data and maintaining ensemble diversity. Higher values
            prefer more diverse ensembles.

            - If None (default): Automatically determine optimal theta using L-curve analysis
            - If float: Use the specified theta value directly

            Default is None (automatic selection).

        theta_range : tuple, list or np.ndarray, optional
            Theta values for the automatic scan. Only used when theta=None.

            - A tuple of length 2, (min_theta, max_theta): theta_n_points
              log-spaced values between the two (positive) endpoints.
            - A list or numpy array: the explicit grid of theta values to
              scan (at least 2 values, all positive); theta_n_points is
              ignored.

            Default is (0.01, 10.0).

        theta_n_points : int, optional
            Number of theta values to test during automatic scan.
            Only used when theta=None and theta_range is a tuple.
            Default is 15.

        max_iterations : int, optional
            Maximum number of optimization iterations. Default is 50000.

        optimizer : str, optional
            Optimization method to use. Default is 'L-BFGS-B'.

        initial_weights : np.ndarray, optional
            Initial weights for ensemble frames. If None, uniform weights are used.

        force_recompute : bool, optional
            If True, forces recomputation even if a cached result for identical
            inputs exists. Default is False.

        verbose : bool, optional
            If True, print optimization progress. Default is True.

        save_theta_scan_plot : str, optional
            Path to save theta scan plot (only used when theta=None / auto mode).
            If None, plot is not saved. Default is None.

        show : bool, optional
            If True and theta=None (auto mode), display the theta-scan plot.
            Default is False.

        theta_method : str, optional
            Method to select optimal theta during the scan (only used when theta=None).
            Options:

            - 'perpendicular' (default): knee by max perpendicular distance to chord
            - 'curvature': knee by Menger curvature

        Returns
        -------
        BMEResult
            Object containing optimization results including weights, chi-squared,
            and convergence information. When theta=None, the details of the
            automatic selection are stored in result.metadata['theta_scan_result'].
            If the optimization failed (result.success is False), a
            RuntimeWarning is issued, the result holds the initial weights, and
            it is NOT cached, so use_bme_weights=True stays unavailable until a
            fit succeeds.

        Raises
        ------
        ValueError
            If calculated_values does not have one row per conformation, or if
            theta_method or theta_range is invalid.
        RuntimeError
            If theta=None and every fit in the theta scan failed.

        Examples
        --------
        >>> from starling.structure.bme import ExperimentalObservable
        >>>
        >>> # Example 1: Automatic theta selection (recommended)
        >>> obs1 = ExperimentalObservable(25.0, 2.0, name="Rg")
        >>> obs2 = ExperimentalObservable(70.0, 5.0, constraint="upper", name="ETE")
        >>>
        >>> rg_values = ensemble.radius_of_gyration()
        >>> ete_values = ensemble.end_to_end_distance()
        >>> calculated = np.column_stack([rg_values, ete_values])
        >>>
        >>> # Automatically finds and uses optimal theta
        >>> result = ensemble.reweight_bme([obs1, obs2], calculated)
        >>> print(f"Auto-selected theta: {result.metadata['theta_used']}")
        >>>
        >>> # Example 2: Manual theta specification
        >>> result = ensemble.reweight_bme([obs1, obs2], calculated, theta=0.5)
        >>>
        >>> # Example 3: Custom theta scan range
        >>> result = ensemble.reweight_bme(
        ...     [obs1, obs2], calculated,
        ...     theta=None,  # auto mode
        ...     theta_range=(0.1, 5.0),
        ...     theta_n_points=20,
        ...     save_theta_scan_plot="my_theta_scan.png"
        ... )
        >>>
        >>> # Now use BME weights in other calculations
        >>> reweighted_rg = ensemble.radius_of_gyration(use_bme_weights=True)

        Notes
        -----
        **Automatic Theta Selection (theta=None)**

        When theta=None, the method performs an L-curve analysis to find the
        optimal theta that balances fit quality (Chi squared) and ensemble diversity (phi).
        The "knee" point of the L-curve is found with the method chosen by
        theta_method (by default the point with the maximum perpendicular
        distance to the line joining the end points of the curve).

        The automatic selection:

        1. Tests multiple theta values across the specified range or grid
        2. Computes Chi squared and phi for each theta
        3. Finds the optimal balance using L-curve analysis, using only the
           fits that succeeded (failed fits are reported with a warning)
        4. Uses the optimal theta for final reweighting
        5. Stores scan results in result.metadata['theta_scan_result']

        **Manual Theta Selection (theta=<value>)**

        When you specify theta explicitly, no scan is performed and the method
        uses your value directly. This is faster but requires you to choose
        theta appropriately for your system.

        **Which Should You Use?**

        - **Use theta=None (auto)** for most applications, especially when:

          * You're unsure what theta value to use
          * You want reproducible, principled parameter selection
          * You're willing to wait ~15x longer by default (one BME fit per
            scanned theta value, theta_n_points=15, instead of 1)

        - **Use theta=<value> (manual)** when:

          * You've already determined an appropriate theta value
          * You need fast reweighting (e.g., in a loop)
          * You're doing sensitivity analysis with specific theta values
        """
        # Validate calculated_values shape
        if calculated_values.shape[0] != self.number_of_conformations:
            raise ValueError(
                f"Number of frames in calculated_values ({calculated_values.shape[0]}) "
                f"must match ensemble size ({self.number_of_conformations})"
            )

        # Reuse the cached result only if it was computed from identical inputs
        bme_cache_key = _bme_cache_key(
            observables,
            calculated_values,
            theta,
            theta_range,
            theta_n_points,
            theta_method,
            max_iterations,
            optimizer,
            initial_weights,
        )
        if self.__bme_result is not None and not force_recompute:
            if bme_cache_key == self.__bme_cache_key:
                if verbose:
                    print("Using cached BME result. Set force_recompute=True to recompute.")
                return self.__bme_result
            if verbose:
                print("BME inputs differ from the cached result; recomputing.")

        # drop any previous result first, so weights from different inputs can
        # never be used if this fit fails
        self.__bme_result = None
        self.__bme_cache_key = None

        # Construct BME object (theta is handled in fit, not here)
        self.__bme = BME(
            observables=observables,
            calculated_values=calculated_values,
            initial_weights=initial_weights,
        )

        # ------------------------------------------------------------------
        # AUTOMATIC THETA MODE (theta is None) -> auto_theta=True
        # ------------------------------------------------------------------
        # Validate theta_method early (applies only to auto-theta mode)
        if theta is None:
            method_lower = str(theta_method).strip().lower()
            if method_lower not in ("perpendicular", "curvature"):
                raise ValueError(
                    f"theta_method must be 'perpendicular' or 'curvature', got: {theta_method}"
                )

            # a tuple is a (min, max) range and a list/array an explicit grid;
            # build it once so what we report is exactly what gets scanned
            theta_grid = build_theta_grid(theta_range, theta_n_points, log_scale=True)

            if verbose:
                print("\n🔍 AUTOMATIC THETA SELECTION MODE")
                print("=" * 60)
                print("Performing L-curve analysis to find optimal theta...")
                print(f"  Range: {theta_grid.min():.4f} to {theta_grid.max():.4f}")
                print(f"  Points: {len(theta_grid)}")
                print(f"  Selection method: {method_lower}")
                print("")

            # Configure how the internal theta scan should run
            theta_scan_kwargs: ThetaScanKwargs = dict(
                theta_range=theta_grid,
                n_points=theta_n_points,
                log_scale=True,
                max_iterations=max_iterations,
                optimizer=optimizer,
                verbose=False,
                progress_callback=(
                    lambda c, t, th: print(f"  Progress: {c}/{t} (θ={th:.4f})")
                    if verbose and (c == 1 or c == t or c % 5 == 0)
                    else None
                ),
                method=method_lower,  # forward the selection method
            )

            # Let BME handle the scan + optimal theta selection
            result = self.__bme.fit(
                max_iterations=max_iterations,
                optimizer=optimizer,
                verbose=verbose,
                theta=None,
                auto_theta=True,
                theta_scan_kwargs=theta_scan_kwargs,
            )

            # Fetch the scan result from BME
            scan_result = self.theta_scan_result

            if scan_result is not None:
                theta_to_use = float(scan_result.optimal_theta)

                if verbose:
                    print("")
                    scan_result.print_summary()

                # Decide whether to save and/or show
                if save_theta_scan_plot is not None or bool(show):
                    scan_result.plot(
                        save_path=save_theta_scan_plot,
                        show=bool(show),
                    )
                    if save_theta_scan_plot is not None and verbose:
                        print(f"\nSaved theta scan plot: {save_theta_scan_plot}")

                # Attach scan metadata
                result.metadata["theta_scan_result"] = scan_result
                result.metadata["theta_selection_method"] = method_lower
                result.metadata["theta_used"] = theta_to_use

                if verbose:
                    print(f"\nUsing optimal theta = {theta_to_use:.4f}")
                    print(f"  Chi squared: {result.chi_squared_final:.4f}")
                    print(f"  phi:    {result.phi:.4f}")
                    print("=" * 60)

            else:
                # Should be rare; be defensive
                if verbose:
                    print(
                        "Warning: auto-theta fit completed but no theta_scan_result found."
                    )
                result.metadata.setdefault("theta_selection_method", "automatic")
                result.metadata.setdefault("theta_used", float(result.theta))

        # ------------------------------------------------------------------
        # MANUAL THETA MODE (theta is not None) -> auto_theta=False
        # ------------------------------------------------------------------
        else:
            if verbose:
                print(f"\n🎯 MANUAL THETA MODE: θ = {theta}")
                print("=" * 60)

            result = self.__bme.fit(
                max_iterations=max_iterations,
                optimizer=optimizer,
                verbose=verbose,
                theta=float(theta),
                auto_theta=False,
            )

            result.metadata["theta_selection_method"] = "manual"
            result.metadata["theta_used"] = float(theta)

        # never cache a failed fit: its weights are just the initial weights
        if not result.success:
            warnings.warn(
                f"BME optimization failed (theta={result.theta}): {result.message}. "
                "The returned result holds the initial weights and has not been cached, "
                "so use_bme_weights=True is unavailable until a fit succeeds.",
                RuntimeWarning,
                stacklevel=2,
            )
            return result

        # Cache result together with the inputs that produced it
        self.__bme_result = result
        self.__bme_cache_key = bme_cache_key
        return result

    def view_theta_scan(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (18, 10),
    ):
        """
        Visualize the most recent theta scan diagnostics for this Ensemble.

        This wraps ThetaScanResult.plot() from the last BME fit that used
        automatic theta selection.

        Parameters
        ----------
        save_path : str, optional
            Path to save the figure. If None, the figure is not written to disk.
        show : bool
            If True, display the figure. Default is True.
        figsize : tuple
            Figure size (width, height). Default is (18, 10).

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.

        Raises
        ------
        ValueError
            If no theta scan has been performed yet.
        """
        scan = self.theta_scan_result
        if scan is None:
            raise ValueError(
                "No theta scan result available. "
                "Run reweight_bme(..., theta=None) at least once to generate a scan."
            )

        fig = scan.plot(figsize=figsize, save_path=save_path, show=show)
        return fig

    @property
    def has_bme_weights(self):
        """
        Check if BME reweighting has been performed.

        Returns
        -------
        bool
            True if BME reweighting has been performed, False otherwise.
        """
        return self.__bme_result is not None and self.__bme_result.success

    @property
    def bme_result(self):
        """
        Get the cached BME result.

        Returns
        -------
        BMEResult or None
            The BME optimization result, or None if reweight_bme() has not been called.
        """
        return self.__bme_result

    @property
    def theta_scan_result(self):
        """
        Theta scan result from the most recent BME fit (if auto-theta was used).

        Returns
        -------
        ThetaScanResult or None
        """
        if self.__bme is None:
            return None

        scan = getattr(self.__bme, "theta_scan_result", None)

        return scan

    def _get_weights(self, use_bme_weights=False):
        """
        Internal method to get weights for calculations.

        Parameters
        ----------
        use_bme_weights : bool
            If True, return BME-optimized weights. If False, return uniform weights.

        Returns
        -------
        np.ndarray
            Array of weights for each frame.
        """
        if use_bme_weights:
            if not self.has_bme_weights:
                raise ValueError(
                    "BME weights requested but no successful BME reweighting is available. "
                    "Call reweight_bme() first."
                )
            assert self.__bme_result is not None
            return self.__bme_result.weights
        else:
            return np.ones(self.number_of_conformations) / self.number_of_conformations


## ------------------------------------------ END OF CLASS DEFINITION


def _array_snapshot(values) -> tuple:
    """
    Return a hashable, copied snapshot of an array-like.

    Parameters
    ----------
    values : array-like
        Values to snapshot (converted with np.asarray).

    Returns
    -------
    tuple
        (shape, dtype string, raw bytes). Two snapshots compare equal only if
        the arrays have identical shape, dtype and contents; because the bytes
        are copied, later in-place edits to the original array do not change
        the snapshot.
    """
    array = np.asarray(values)
    return (array.shape, array.dtype.str, array.tobytes())


def _bme_cache_key(
    observables: list,
    calculated_values: np.ndarray,
    theta: Optional[float],
    theta_range: Union[Tuple[float, float], List[float], np.ndarray],
    theta_n_points: int,
    theta_method: str,
    max_iterations: int,
    optimizer: str,
    initial_weights: Optional[np.ndarray],
) -> tuple:
    """
    Build a key recording every input that determines a BME reweighting result.

    Ensemble.reweight_bme() stores this key next to its cached result and only
    reuses that result when a later call produces an identical key, so a call
    with new observables, calculated values or theta settings is never answered
    with stale weights.

    Parameters
    ----------
    observables : list[ExperimentalObservable]
        Experimental observables; compared through their repr, which lists
        value, uncertainty, constraint and name.
    calculated_values : np.ndarray
        Per-frame calculated observables, shape (n_frames, n_observables).
    theta : float or None
        Manual theta, or None for automatic selection.
    theta_range : tuple, list or np.ndarray
        Theta scan range (tuple) or explicit grid (list/array). Only part of
        the key when theta is None.
    theta_n_points : int
        Number of scan points for a tuple range. Only part of the key when
        theta is None.
    theta_method : str
        Knee selection method. Only part of the key when theta is None.
    max_iterations : int
        Maximum optimizer iterations.
    optimizer : str
        Optimizer name.
    initial_weights : np.ndarray or None
        Initial frame weights, or None for uniform weights.

    Returns
    -------
    tuple
        Hashable key; equal keys mean identical BME inputs.
    """
    theta_settings: tuple
    if theta is None:
        # a tuple is a range and a list/array an explicit grid, so the type matters
        theta_settings = (
            None,
            isinstance(theta_range, tuple),
            _array_snapshot(theta_range),
            int(theta_n_points),
            str(theta_method).strip().lower(),
        )
    else:
        theta_settings = (float(theta),)

    if initial_weights is None:
        initial_weights_snapshot = None
    else:
        initial_weights_snapshot = _array_snapshot(initial_weights)

    return (
        repr(observables),
        _array_snapshot(calculated_values),
        theta_settings,
        int(max_iterations),
        str(optimizer),
        initial_weights_snapshot,
    )


def load_ensemble(filename, ignore_structures=False):
    """
    Function to read in a STARLING ensemble from a file and return the
    STARLING ensemble object.

    Parameters
    ---------------
    filename : str
        The filename to read the ensemble from (should be a .starling
        file generated by STARLING)

    ignore_structures : bool
        If set to True, the function will discard structures that
        are part of the .starling file, which saves memory if you
        don't need them. The whole file is still read, so this does
        not make loading faster.
    """

    # note there's exception handling in the utilities.py file, and we automatically
    # detect the compression algorithm based on the file extension
    return_dict = utilities.read_starling_ensemble(filename)

    # make sure we can extract out the core components
    try:
        sequence = return_dict["sequence"]
        distance_maps = return_dict["distance_maps"]
        if ignore_structures:
            traj = None
        else:
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
        E = Ensemble(
            distance_maps,
            sequence,
            traj,
            ionic_strength=return_dict.get("IONIC_STRENGTH"),
        )
    except Exception as e:
        raise Exception(
            f"Error initializing STARLING ensemble: {filename} [error 3]; error {e}"
        )

    # finally we over-write the metadata
    E.metadata["DEFAULT_ENCODER_WEIGHTS_PATH"] = DEFAULT_ENCODER_WEIGHTS_PATH
    E.metadata["DEFAULT_DDPM_WEIGHTS_PATH"] = DEFAULT_DDPM_WEIGHTS_PATH
    E.metadata["VERSION"] = VERSION
    E.metadata["DATE"] = DATE

    return E
