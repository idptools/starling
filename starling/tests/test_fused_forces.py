"""Correctness checks for the optional CUDA pair-force kernel."""

import pytest
import torch

pytest.importorskip("triton")

from starling.minimizer import DistanceRestraints, MpipiGG
from starling.minimizer.fused_forces import fused_forces


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("length", [1, 2, 7, 129])
@pytest.mark.parametrize(("residue", "ionic_strength"), [("D", 0), ("G", 150)])
def test_fused_forces_match_eager_for_batches_and_partial_tiles(length, residue, ionic_strength):
    sequence = residue * length
    forcefield = MpipiGG(sequence, ionic_strength=ionic_strength, device="cuda", dtype=torch.float32)
    positions = torch.arange(length, device="cuda", dtype=torch.float32)
    coordinates = torch.stack((3.81 * positions, positions.sin(), positions.cos()), dim=-1)
    coordinates = torch.stack((coordinates, coordinates + 0.1 * positions.sin()[:, None])).contiguous()
    reference = torch.cdist(coordinates, coordinates, compute_mode="donot_use_mm_for_euclid_dist")
    reference *= torch.tensor([0.7, 1.3], device="cuda")[:, None, None]
    restraints = DistanceRestraints(reference, sigma=forcefield._sigma, device="cuda", dtype=torch.float32)
    force = fused_forces(forcefield, restraints)
    indices = torch.tensor([1, 0], device="cuda")

    for batch, frames in ((coordinates, indices), (coordinates[:1], indices[:1])):
        expected = forcefield.evaluate(batch, restraints, frames).forces
        actual = force(batch, frames)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_float32_fused_forces_remain_finite_at_severe_clashes():
    forcefield = MpipiGG("DGDGDGD", device="cuda", dtype=torch.float32)
    coordinates = torch.zeros((2, 7, 3), device="cuda")
    positions = torch.arange(7, device="cuda")
    coordinates[:, :, 0] = positions * 0.1 + positions.square() * 0.01
    reference = torch.cdist(coordinates, coordinates, compute_mode="donot_use_mm_for_euclid_dist")
    restraints = DistanceRestraints(reference, sigma=forcefield._sigma, device="cuda", dtype=torch.float32)
    frames = torch.tensor([1, 0], device="cuda")

    actual = fused_forces(forcefield, restraints)(coordinates, frames)
    expected = forcefield.evaluate(coordinates, restraints, frames).forces
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-2)
