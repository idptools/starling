"""
Unit and regression test for the starling package.
"""

# Import package, test suite, and other packages as needed
import sys
import numpy as np

import pytest

import starling

from starling import generate
from starling.structure.ensemble import Ensemble

from soursop.sstrajectory import SSTrajectory

import torch


def test_starling_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "starling" in sys.modules


def test_ensemble_generation():

    # define sequence
    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'

    
    C = generate(seq,conformations=100, verbose=False, show_progress_bar=False, return_data=True, return_structures=False)
    E=Ensemble(C['sequence_1'], seq)

    assert len(E) == 100
    assert abs(E.radius_of_gyration(return_mean=True) - 32) < 2.5
    assert abs(E.end_to_end_distance(return_mean=True) - 85) < 6
    assert E.sequence == seq

    # check we can build a 
    t = E.ensemble_trajectory()
    np.isclose(np.mean(t.get_radius_of_gyration()) , E.radius_of_gyration(return_mean=True), rtol=0.01, atol=0.01)

    # check we can write a trajectory
    E.save_trajectory('outdata/test.pdb', pdb_trajectory=True)



def test_ensemble_generation_cpu():

    # define sequence
    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'

    
    C = generate(seq,conformations=100, verbose=False, show_progress_bar=False, return_data=True, return_structures=False, device='cpu')
    E=Ensemble(C['sequence_1'], seq)

    assert len(E) == 100
    assert abs(E.radius_of_gyration(return_mean=True) - 32) < 2.5
    assert abs(E.end_to_end_distance(return_mean=True) - 85) < 6
    assert E.sequence == seq

    # check we can build a 
    t = E.ensemble_trajectory(device='cpu')
    np.isclose(np.mean(t.get_radius_of_gyration()) , E.radius_of_gyration(return_mean=True), rtol=0.01, atol=0.01)

    # check we can write a trajectory
    E.save_trajectory('outdata/test.pdb', pdb_trajectory=True)



def test_ensemble_generation_mps():

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    # define sequence
    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'
    
    C = generate(seq,conformations=100, verbose=False, show_progress_bar=False, return_data=True, return_structures=False, device='mps')
    
    E = Ensemble(C['sequence_1'], seq)

    assert len(E) == 100
    assert abs(E.radius_of_gyration(return_mean=True) - 32) < 2.5
    assert abs(E.end_to_end_distance(return_mean=True) - 85) < 6
    assert E.sequence == seq

    # check we can build a 
    t = E.ensemble_trajectory(device='mps')
    np.isclose(np.mean(t.get_radius_of_gyration()) , E.radius_of_gyration(return_mean=True), rtol=0.01, atol=0.01)

    # check we can write a trajectory
    E.save_trajectory('outdata/test.pdb', pdb_trajectory=True)


def test_ensemble_generation_cuda():

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        

    # define sequence
    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'
    
    
    C = generate(seq,conformations=100, verbose=False, show_progress_bar=False, return_data=True, return_structures=False, device='cuda')
            
    E=Ensemble(C['sequence_1'], seq)

    assert len(E) == 100
    assert abs(E.radius_of_gyration(return_mean=True) - 32) < 2.5
    assert abs(E.end_to_end_distance(return_mean=True) - 85) < 6
    assert E.sequence == seq

    # check we can build a 
    t = E.ensemble_trajectory(device='cuda')
    np.isclose(np.mean(t.get_radius_of_gyration()) , E.radius_of_gyration(return_mean=True), rtol=0.01, atol=0.01)

    # check we can write a trajectory
    E.save_trajectory('outdata/test.pdb', pdb_trajectory=True)
    

def test_ensemble_generation_gd():

    # define sequence
    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'

    
    C = generate(seq,conformations=5, verbose=False, show_progress_bar=False, return_data=True, return_structures=False)
    E=Ensemble(C['sequence_1'], seq)
    # check we can build a 
    t = E.ensemble_trajectory(method='gd')
    np.isclose(np.mean(t.get_radius_of_gyration()) , E.radius_of_gyration(return_mean=True), rtol=0.01, atol=0.01)

    # check we can write a trajectory
    E.save_trajectory('outdata/test.pdb', pdb_trajectory=True)


def test_ensemble_reconstruction():
    seq = 'ASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAPSPASPASPAPSPASPAPSPPASPASPAASAPASPAPSPAP'
    C = generate(seq,conformations=100, verbose=False, show_progress_bar=False, return_data=True, return_structures=True)
    
    E=Ensemble(C['sequence_1'], seq)
    p = SSTrajectory(TRJ=C['sequence_1_traj']).proteinTrajectoryList[0]
    
    assert np.all(np.isclose(p.get_end_to_end_distance(), E.end_to_end_distance(), atol=1, rtol=1))
    
