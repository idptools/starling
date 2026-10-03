"""
Unit and regression test for the starling package.
"""

# Import package, test suite, and other packages as needed
import sys
import numpy as np

import pytest


from starling import generate, load_ensemble
from starling.structure.ensemble import Ensemble
from starling import utilities

import torch


def test_starling_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "starling" in sys.modules


@pytest.mark.parametrize("compression", [None, "lzma", "gzip"])
@pytest.mark.parametrize("reduce_precision", [False, True])
def test_ensemble_generation_save_and_load(tmp_path, compression, reduce_precision):
    seq = "ACDEF"
    xyz = np.random.default_rng(0).normal(size=(10, len(seq), 3))
    maps = np.linalg.norm(xyz[:, :, None] - xyz[:, None, :], axis=-1)
    ensemble = Ensemble(maps, seq)
    path = str(tmp_path / "ensemble")
    ensemble.save(
        path,
        compress=compression is not None,
        compression_algorithm=compression or "lzma",
        reduce_precision=reduce_precision,
    )
    suffix = {None: ".starling", "lzma": ".starling.xz", "gzip": ".starling.gzip"}[compression]
    loaded = load_ensemble(path + suffix)
    assert loaded.sequence == seq
    assert len(loaded) == len(ensemble)
    if reduce_precision:
        # Rounding to one decimal contributes at most 0.05 A; float16 adds
        # at most 0.002 A for these distances (all below 8 A).
        np.testing.assert_allclose(loaded.distance_maps(), maps, atol=0.052, rtol=0)
    else:
        np.testing.assert_array_equal(loaded.distance_maps(), maps)


@pytest.mark.slow
def test_ensemble_generation(tmp_path):

    # define sequence
    seq = "ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP"

    C = generate(
        seq, conformations=100, verbose=False, show_progress_bar=False, return_data=True, return_structures=False
    )
    E = C["sequence_1"]

    assert len(E) == 100
    assert abs(E.radius_of_gyration(return_mean=True) - 32) < 3
    assert abs(E.end_to_end_distance(return_mean=True) - 85) < 8
    assert E.sequence == seq

    # check we can build a
    t = E.trajectory
    assert np.isclose(
        np.mean(t.get_radius_of_gyration()), E.radius_of_gyration(return_mean=True), rtol=0.01, atol=0.01
    )

    # check we can write a trajectory
    E.save_trajectory(str(tmp_path / "test"), pdb_trajectory=True)


@pytest.mark.slow
def test_ensemble_generation_single_ensemble(tmp_path):

    # define sequence
    seq = "ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP"

    # check we can get a single Ensembe object
    E = generate(
        seq,
        conformations=100,
        verbose=False,
        show_progress_bar=False,
        return_data=True,
        return_structures=False,
        return_single_ensemble=True,
    )

    assert len(E) == 100
    assert abs(E.radius_of_gyration(return_mean=True) - 32) < 3
    assert abs(E.end_to_end_distance(return_mean=True) - 85) < 8
    assert E.sequence == seq

    # check we can build a
    t = E.trajectory
    assert np.isclose(
        np.mean(t.get_radius_of_gyration()), E.radius_of_gyration(return_mean=True), rtol=0.01, atol=0.01
    )

    # check we can write a trajectory
    E.save_trajectory(str(tmp_path / "test"), pdb_trajectory=True)


@pytest.mark.slow
def test_ensemble_generation_cpu(tmp_path):

    # define sequence
    seq = "ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP"

    C = generate(
        seq,
        conformations=100,
        verbose=False,
        show_progress_bar=False,
        return_data=True,
        return_structures=False,
        device="cpu",
    )
    E = C["sequence_1"]

    assert len(E) == 100
    assert abs(E.radius_of_gyration(return_mean=True) - 32) < 3
    assert abs(E.end_to_end_distance(return_mean=True) - 85) < 8
    assert E.sequence == seq

    # check we can build a
    t = E.build_ensemble_trajectory(device="cpu")
    assert np.isclose(
        np.mean(t.get_radius_of_gyration()), E.radius_of_gyration(return_mean=True), rtol=0.01, atol=0.01
    )

    # check we can write a trajectory
    E.save_trajectory(str(tmp_path / "test"), pdb_trajectory=True)


@pytest.mark.slow
def test_ensemble_generation_mps(tmp_path):

    if not torch.backends.mps.is_available():
        raise pytest.skip.Exception("MPS is not available")

    # define sequence
    seq = "ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP"

    C = generate(
        seq,
        conformations=100,
        verbose=False,
        show_progress_bar=False,
        return_data=True,
        return_structures=False,
        device="mps",
    )

    E = C["sequence_1"]

    assert len(E) == 100
    assert abs(E.radius_of_gyration(return_mean=True) - 32) < 3
    assert abs(E.end_to_end_distance(return_mean=True) - 85) < 8
    assert E.sequence == seq

    # check we can build a
    t = E.build_ensemble_trajectory(device="mps")
    assert np.isclose(
        np.mean(t.get_radius_of_gyration()), E.radius_of_gyration(return_mean=True), rtol=0.01, atol=0.01
    )

    # check we can write a trajectory
    E.save_trajectory(str(tmp_path / "test"), pdb_trajectory=True)


@pytest.mark.slow
def test_ensemble_generation_cuda(tmp_path):

    if not torch.cuda.is_available():
        raise pytest.skip.Exception("CUDA is not available")

    # define sequence
    seq = "ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP"

    C = generate(
        seq,
        conformations=100,
        verbose=False,
        show_progress_bar=False,
        return_data=True,
        return_structures=False,
        device="cuda",
    )

    E = C["sequence_1"]

    assert len(E) == 100
    assert abs(E.radius_of_gyration(return_mean=True) - 32) < 3
    assert abs(E.end_to_end_distance(return_mean=True) - 85) < 8
    assert E.sequence == seq

    # check we can build a
    t = E.build_ensemble_trajectory(device="cuda")
    assert np.isclose(
        np.mean(t.get_radius_of_gyration()), E.radius_of_gyration(return_mean=True), rtol=0.01, atol=0.01
    )

    # check we can write a trajectory
    E.save_trajectory(str(tmp_path / "test"), pdb_trajectory=True)


@pytest.mark.slow
def test_ensemble_reconstruction_re():
    seq = "ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP"
    C = generate(
        seq, conformations=100, verbose=False, show_progress_bar=False, return_data=True, return_structures=True
    )

    E = C["sequence_1"]
    p = E.trajectory

    # absolute tollerance of 10 Angstroms
    assert np.all(np.isclose(p.get_end_to_end_distance(), E.end_to_end_distance(), atol=10, rtol=0))


@pytest.mark.slow
def test_ensemble_reconstruction_dm():
    #
    # MDS reconstruction is hard, so our tolerance here is ~5% of frames can have ONE OR MORE distance that
    # is off by more than 10 Angstroms. This translates to a very low average error, but we do want to do
    # the full comparison to get a sense of the 'true' error here...
    #

    seq = "ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP"
    n_confs = 100
    E = generate(
        seq,
        conformations=n_confs,
        verbose=False,
        show_progress_bar=False,
        return_single_ensemble=True,
        return_structures=True,
    )

    P = E.trajectory

    # all structural distance maps ("reconstruction")
    A = utilities.symmetrize_distance_maps(P.get_distance_map(return_instantaneous_maps=True, verbose=False)[0])

    # all STARLING distance maps ("truth")
    B = E.distance_maps()

    # the code below finds the number of frames where 1 or more i-j distances are off by more than 10 angstroms
    mask = np.isclose(A, B, atol=10, rtol=0)  # Boolean array
    offending_indices = np.where(~mask)  # Indices where values are NOT clos
    bad_frames = len(set(offending_indices[0]))

    # 5% of frames are allowed to be bad
    assert bad_frames <= (1 + int(n_confs) * 0.05)


@pytest.mark.slow
def test_ensemble_reconstruction_dm_CPU():
    #
    # MDS reconstruction is hard, so our tolerance here is ~10% of frames can have ONE OR MORE distance that
    # is off by more than 10 Angstroms. This translates to a very low average error, but we do want to do
    # the full comparison to get a sense of the 'true' error here...
    #

    seq = "ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP"
    n_confs = 100
    E = generate(seq, conformations=n_confs, verbose=False, show_progress_bar=False, return_single_ensemble=True)

    # specifically build with CPU regardless of what was used for generation...
    E.build_ensemble_trajectory(device="cpu")

    P = E.trajectory

    # all structural distance maps ("reconstruction")
    A = utilities.symmetrize_distance_maps(P.get_distance_map(return_instantaneous_maps=True, verbose=False)[0])

    # all STARLING distance maps ("truth")
    B = E.distance_maps()

    # the code below finds the number of frames where 1 or more i-j distances are off by more than 10 angstroms
    mask = np.isclose(A, B, atol=10, rtol=0)  # Boolean array
    offending_indices = np.where(~mask)  # Indices where values are NOT clos
    bad_frames = len(set(offending_indices[0]))

    # ~10% of frames are allowed to be bad
    assert bad_frames <= (1 + int(n_confs) * 0.10)


@pytest.mark.slow
def test_ensemble_reconstruction_dm_mps():
    #
    # MDS reconstruction is hard, so our tolerance here is ~10% of frames can have ONE OR MORE distance that
    # is off by more than 10 Angstroms. This translates to a very low average error, but we do want to do
    # the full comparison to get a sense of the 'true' error here...
    #
    if not torch.backends.mps.is_available():
        raise pytest.skip.Exception("MPS is not available")

    seq = "ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP"
    n_confs = 100
    E = generate(seq, conformations=n_confs, verbose=False, show_progress_bar=False, return_single_ensemble=True)

    # specifically build with CPU regardless of what was used for generation...
    E.build_ensemble_trajectory(device="mps")

    P = E.trajectory

    # all structural distance maps ("reconstruction")
    A = utilities.symmetrize_distance_maps(P.get_distance_map(return_instantaneous_maps=True, verbose=False)[0])

    # all STARLING distance maps ("truth")
    B = E.distance_maps()

    # the code below finds the number of frames where 1 or more i-j distances are off by more than 10 angstroms
    mask = np.isclose(A, B, atol=10, rtol=0)  # Boolean array
    offending_indices = np.where(~mask)  # Indices where values are NOT clos
    bad_frames = len(set(offending_indices[0]))

    # ~10% of frames are allowed to be bad
    assert bad_frames <= (1 + int(n_confs) * 0.10)


@pytest.mark.slow
def test_ensemble_reconstruction_dm_cuda():
    #
    # MDS reconstruction is hard, so our tolerance here is ~10% of frames can have ONE OR MORE distance that
    # is off by more than 10 Angstroms. This translates to a very low average error, but we do want to do
    # the full comparison to get a sense of the 'true' error here...
    #
    if not torch.cuda.is_available():
        raise pytest.skip.Exception("CUDA is not available")

    seq = "ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP"
    n_confs = 100
    E = generate(seq, conformations=n_confs, verbose=False, show_progress_bar=False, return_single_ensemble=True)

    # specifically build with CPU regardless of what was used for generation...
    E.build_ensemble_trajectory(device="cuda")

    P = E.trajectory

    # all structural distance maps ("reconstruction")
    A = utilities.symmetrize_distance_maps(P.get_distance_map(return_instantaneous_maps=True, verbose=False)[0])

    # all STARLING distance maps ("truth")
    B = E.distance_maps()

    # the code below finds the number of frames where 1 or more i-j distances are off by more than 10 angstroms
    mask = np.isclose(A, B, atol=10, rtol=0)  # Boolean array
    offending_indices = np.where(~mask)  # Indices where values are NOT clos
    bad_frames = len(set(offending_indices[0]))

    # ~10% of frames are allowed to be bad
    assert bad_frames <= (1 + int(n_confs) * 0.10)


@pytest.mark.slow
def test_skip_long_seqs():
    """
    Check we can pass a sequence that's too long and it's skipped but
    does not trigger a total failure.
    """

    seqs = {}
    seqs["a"] = "AP" * 20
    seqs["b"] = "AP" * 200

    C = generate(
        seqs, conformations=10, verbose=False, show_progress_bar=False, return_data=True, return_structures=True
    )
    assert len(C) == 1


def test_invalid_input_options():

    seq = "ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP"

    seqs = {}
    seqs["a"] = "AP" * 20
    seqs["b"] = "AQ" * 30

    # check we fail if return_data=False and output_directory is None
    with pytest.raises(ValueError):
        generate(seq, conformations=10, verbose=False, show_progress_bar=False, return_data=False)

    # check we fail if we pass multiple sequences and request a single ensemble
    with pytest.raises(ValueError):
        generate(seqs, conformations=10, verbose=False, show_progress_bar=False, return_single_ensemble=True)
