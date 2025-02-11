"""
Unit and regression test for the starling package.
"""

# Import package, test suite, and other packages as needed
import sys
import os
import numpy as np

import pytest

import starling

from starling import generate, load_ensemble
from starling.structure.ensemble import Ensemble
from starling import utilities

from soursop.sstrajectory import SSTrajectory

import torch


def test_starling_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "starling" in sys.modules



def test_ensemble_generation_save_and_load():

    # define sequence
    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'

    
    E = generate(seq,conformations=10, verbose=False, show_progress_bar=False, return_data=True, return_structures=False, return_single_ensemble=True)
    assert len(E) == 10

    # check we can write a trajectory and load it uncompressed
    E.save('outdata/test_uncompressed')
    E_test_uncompressed = load_ensemble('outdata/test_uncompressed.starling')
    assert np.all(E_test_uncompressed.end_to_end_distance() == E.end_to_end_distance())
    os.remove('outdata/test_uncompressed.starling')

    ##
    ## COMPRESSION TESTS
    ## 

    # check we can write a trajectory using default compression and load 
    E.save('outdata/test_compressed', compress=True)
    E_test_compressed = load_ensemble('outdata/test_compressed.starling.xz')
    assert np.all(np.isclose(E_test_compressed.end_to_end_distance(), E.end_to_end_distance(), atol=0.1))
    os.remove('outdata/test_compressed.starling.xz')

    # check we can write a trajectory using default lzma explicitly
    E.save('outdata/test_compressed', compress=True, compression_algorithm='lzma')
    E_test_compressed = load_ensemble('outdata/test_compressed.starling.xz')
    assert np.all(np.isclose(E_test_compressed.end_to_end_distance(), E.end_to_end_distance(), atol=0.1))
    os.remove('outdata/test_compressed.starling.xz')

    # check we can write a trajectory using default gzip explicitly
    E.save('outdata/test_compressed', compress=True, compression_algorithm='gzip')
    E_test_compressed = load_ensemble('outdata/test_compressed.starling.gzip')
    assert np.all(np.isclose(E_test_compressed.end_to_end_distance(), E.end_to_end_distance(), atol=0.1))
    os.remove('outdata/test_compressed.starling.gzip')    


    ## now check we can do compresion with reduce precision explicitly to False (means should be lossless)

    # check we can write a trajectory using default compression and load 
    E.save('outdata/test_compressed', compress=True, reduce_precision=False)
    E_test_compressed = load_ensemble('outdata/test_compressed.starling.xz')
    assert np.all(E_test_compressed.end_to_end_distance() == E.end_to_end_distance())
    os.remove('outdata/test_compressed.starling.xz')

    # check we can write a trajectory using default lzma explicitly
    E.save('outdata/test_compressed', compress=True, compression_algorithm='lzma',  reduce_precision=False)
    E_test_compressed = load_ensemble('outdata/test_compressed.starling.xz')
    assert np.all(E_test_compressed.end_to_end_distance() == E.end_to_end_distance())
    os.remove('outdata/test_compressed.starling.xz')

    # check we can write a trajectory using default gzip explicitly
    E.save('outdata/test_compressed', compress=True, compression_algorithm='gzip',  reduce_precision=False)
    E_test_compressed = load_ensemble('outdata/test_compressed.starling.gzip')
    assert np.all(E_test_compressed.end_to_end_distance() == E.end_to_end_distance())
    os.remove('outdata/test_compressed.starling.gzip')    

    ## now check we can do compresion with reduce precision explicitly to True (again should be lossy)

    # check we can write a trajectory using default compression and load 
    E.save('outdata/test_compressed', compress=True, reduce_precision=False)
    E_test_compressed = load_ensemble('outdata/test_compressed.starling.xz')
    assert np.all(np.isclose(E_test_compressed.end_to_end_distance(), E.end_to_end_distance(), atol=0.1))
    os.remove('outdata/test_compressed.starling.xz')

    # check we can write a trajectory using default lzma explicitly
    E.save('outdata/test_compressed', compress=True, compression_algorithm='lzma',  reduce_precision=False)
    E_test_compressed = load_ensemble('outdata/test_compressed.starling.xz')
    assert np.all(np.isclose(E_test_compressed.end_to_end_distance(), E.end_to_end_distance(), atol=0.1))
    os.remove('outdata/test_compressed.starling.xz')

    # check we can write a trajectory using default gzip explicitly
    E.save('outdata/test_compressed', compress=True, compression_algorithm='gzip',  reduce_precision=False)
    E_test_compressed = load_ensemble('outdata/test_compressed.starling.gzip')
    assert np.all(np.isclose(E_test_compressed.end_to_end_distance(), E.end_to_end_distance(), atol=0.1))
    os.remove('outdata/test_compressed.starling.gzip')    




def test_ensemble_generation():

    # define sequence
    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'

    
    C = generate(seq,conformations=100, verbose=False, show_progress_bar=False, return_data=True, return_structures=False)
    E = C['sequence_1']

    assert len(E) == 100
    assert abs(E.radius_of_gyration(return_mean=True) - 32) < 3
    assert abs(E.end_to_end_distance(return_mean=True) - 85) < 8
    assert E.sequence == seq

    # check we can build a trajectory
    t = E.trajectory
    np.isclose(np.mean(t.get_radius_of_gyration()) , E.radius_of_gyration(return_mean=True), rtol=0.01, atol=0.01)


def test_ensemble_generation():

    # define sequence
    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'

    
    C = generate(seq,conformations=100, verbose=False, show_progress_bar=False, return_data=True, return_structures=False)
    E = C['sequence_1']

    assert len(E) == 100
    assert abs(E.radius_of_gyration(return_mean=True) - 32) < 3
    assert abs(E.end_to_end_distance(return_mean=True) - 85) < 8
    assert E.sequence == seq

    # check we can build a 
    t = E.trajectory
    np.isclose(np.mean(t.get_radius_of_gyration()) , E.radius_of_gyration(return_mean=True), rtol=0.01, atol=0.01)

    # check we can write a trajectory
    E.save_trajectory('outdata/test.pdb', pdb_trajectory=True)


def test_ensemble_generation_single_ensemble():

    # define sequence
    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'

    # check we can get a single Ensembe object    
    E = generate(seq,conformations=100, verbose=False, show_progress_bar=False, return_data=True, return_structures=False, return_single_ensemble=True)
    
    assert len(E) == 100
    assert abs(E.radius_of_gyration(return_mean=True) - 32) < 3
    assert abs(E.end_to_end_distance(return_mean=True) - 85) < 8
    assert E.sequence == seq

    # check we can build a 
    t = E.trajectory
    np.isclose(np.mean(t.get_radius_of_gyration()) , E.radius_of_gyration(return_mean=True), rtol=0.01, atol=0.01)

    # check we can write a trajectory
    E.save_trajectory('outdata/test.pdb', pdb_trajectory=True)

def test_ensemble_generation_cpu():

    # define sequence
    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'

    
    C = generate(seq,conformations=100, verbose=False, show_progress_bar=False, return_data=True, return_structures=False, device='cpu')
    E = C['sequence_1']

    assert len(E) == 100
    assert abs(E.radius_of_gyration(return_mean=True) - 32) < 3
    assert abs(E.end_to_end_distance(return_mean=True) - 85) < 8
    assert E.sequence == seq

    # check we can build a 
    t = E.build_ensemble_trajectory(device='cpu')
    np.isclose(np.mean(t.get_radius_of_gyration()) , E.radius_of_gyration(return_mean=True), rtol=0.01, atol=0.01)

    # check we can write a trajectory
    E.save_trajectory('outdata/test.pdb', pdb_trajectory=True)



def test_ensemble_generation_mps():

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    # define sequence
    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'
    
    C = generate(seq,conformations=100, verbose=False, show_progress_bar=False, return_data=True, return_structures=False, device='mps')
    
    E = C['sequence_1']

    assert len(E) == 100
    assert abs(E.radius_of_gyration(return_mean=True) - 32) < 3
    assert abs(E.end_to_end_distance(return_mean=True) - 85) < 8
    assert E.sequence == seq

    # check we can build a 
    t = E.build_ensemble_trajectory(device='mps')
    np.isclose(np.mean(t.get_radius_of_gyration()) , E.radius_of_gyration(return_mean=True), rtol=0.01, atol=0.01)

    # check we can write a trajectory
    E.save_trajectory('outdata/test.pdb', pdb_trajectory=True)


def test_ensemble_generation_cuda():

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        

    # define sequence
    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'
    
    
    C = generate(seq,conformations=100, verbose=False, show_progress_bar=False, return_data=True, return_structures=False, device='cuda')
            
    E = C['sequence_1']

    assert len(E) == 100
    assert abs(E.radius_of_gyration(return_mean=True) - 32) < 3
    assert abs(E.end_to_end_distance(return_mean=True) - 85) < 8
    assert E.sequence == seq

    # check we can build a 
    t = E.build_ensemble_trajectory(device='cuda')
    np.isclose(np.mean(t.get_radius_of_gyration()) , E.radius_of_gyration(return_mean=True), rtol=0.01, atol=0.01)

    # check we can write a trajectory
    E.save_trajectory('outdata/test.pdb', pdb_trajectory=True)
    
def test_ensemble_reconstruction_re():
    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'
    C = generate(seq,conformations=100, verbose=False, show_progress_bar=False, return_data=True, return_structures=True)
    
    E = C['sequence_1']
    p = E.trajectory
    
    # absolute tollerance of 10 Angstroms
    assert np.all(np.isclose(p.get_end_to_end_distance(), E.end_to_end_distance(), atol=10, rtol=0))

def test_ensemble_reconstruction_dm():
    #
    # MDS reconstruction is hard, so our tolerance here is ~5% of frames can have ONE OR MORE distance that 
    # is off by more than 10 Angstroms. This translates to a very low average error, but we do want to do
    # the full comparison to get a sense of the 'true' error here...
    #

    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'
    n_confs = 100
    E = generate(seq,conformations=n_confs, verbose=False, show_progress_bar=False, return_single_ensemble=True, return_structures=True)

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
    assert bad_frames <= (1+int(n_confs)*0.05)  
    

def test_ensemble_reconstruction_dm_CPU():
    #
    # MDS reconstruction is hard, so our tolerance here is ~10% of frames can have ONE OR MORE distance that 
    # is off by more than 10 Angstroms. This translates to a very low average error, but we do want to do
    # the full comparison to get a sense of the 'true' error here...
    #

    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'
    n_confs = 100
    E = generate(seq,conformations=n_confs, verbose=False, show_progress_bar=False, return_single_ensemble=True)

    # specifically build with CPU regardless of what was used for generation...
    E.build_ensemble_trajectory(device='cpu')

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
    assert bad_frames <= (1+int(n_confs)*0.10)  


def test_ensemble_reconstruction_dm_mps():
    #
    # MDS reconstruction is hard, so our tolerance here is ~10% of frames can have ONE OR MORE distance that 
    # is off by more than 10 Angstroms. This translates to a very low average error, but we do want to do
    # the full comparison to get a sense of the 'true' error here...
    #
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'
    n_confs = 100
    E = generate(seq,conformations=n_confs, verbose=False, show_progress_bar=False, return_single_ensemble=True)

    # specifically build with CPU regardless of what was used for generation...
    E.build_ensemble_trajectory(device='mps')

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
    assert bad_frames <= (1+int(n_confs)*0.10)  

def test_ensemble_reconstruction_dm_cuda():
    #
    # MDS reconstruction is hard, so our tolerance here is ~10% of frames can have ONE OR MORE distance that 
    # is off by more than 10 Angstroms. This translates to a very low average error, but we do want to do
    # the full comparison to get a sense of the 'true' error here...
    #
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'
    n_confs = 100
    E = generate(seq,conformations=n_confs, verbose=False, show_progress_bar=False, return_single_ensemble=True)

    # specifically build with CPU regardless of what was used for generation...
    E.build_ensemble_trajectory(device='cuda')

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
    assert bad_frames <= (1+int(n_confs)*0.10)  


def test_skip_long_seqs():
    """
    Check we can pass a sequence that's too long and it's skipped but
    does not trigger a total failure.
    """
    
    seqs = {}
    seqs['a'] = 'AP'*20
    seqs['b'] = 'AP'*200

    C = generate(seqs, conformations=10, verbose=False, show_progress_bar=False, return_data=True, return_structures=True)
    assert len(C) == 1

    
def test_invalid_input_options():

    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'

    seqs = {}
    seqs['a'] = 'AP'*20
    seqs['b'] = 'AQ'*30


    # check we fail if return_data=False and output_directory is None
    with pytest.raises(ValueError):
        generate(seq, conformations=10, verbose=False, show_progress_bar=False, return_data=False)
    
    # check we fail if we pass multiple sequences and request a single ensemble
    with pytest.raises(ValueError):
        generate(seqs, conformations=10, verbose=False, show_progress_bar=False, return_single_ensemble=True)


    
