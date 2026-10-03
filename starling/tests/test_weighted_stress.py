import numpy as np
import json
from pathlib import Path
import pytest
import torch

from starling.structure.coordinates import distance_matrix_to_3d_structure_torch_mds


def test_weights_match_recorded_posterior_sample_calibration():
    from starling.structure.weighted_stress import map_error_weights

    record = json.loads((Path(__file__).parents[2] / "devtools/scripts/weighted_stress_expanded_study.json").read_text())
    model = record["fits"]["sample"]["model"]
    errors = np.array(model["short_separation_error"])
    for n in (61, 151, 327):
        plateau = model["plateau_intercept"] + model["plateau_slope"] * n
        weights = map_error_weights(n).numpy()
        np.testing.assert_allclose(
            weights[0, 1:7], 1 / np.minimum(errors, plateau) ** 2, rtol=1e-6
        )
        np.testing.assert_allclose(weights[0, 7:], 1 / plateau**2, rtol=1e-6)


@pytest.mark.parametrize("as_tensor", [False, True])
def test_smacof_accepts_float64_distance_maps(as_tensor):
    xyz = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]], dtype=np.float64)
    target = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)[None]
    coords, _ = distance_matrix_to_3d_structure_torch_mds(
        torch.from_numpy(target) if as_tensor else target,
        n_iter=3,
        device="cpu",
        progress_bar=False,
    )
    recovered = np.linalg.norm(coords[:, :, None] - coords[:, None, :], axis=-1)
    np.testing.assert_allclose(recovered, target, atol=1e-5)


@pytest.mark.parametrize("n_points", [1, 2, 5])
def test_smacof_starts_with_exact_three_dimensional_geometry(n_points):
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0],
            [2.0, 3.0, 4.0],
        ]
    )[:n_points]
    target = torch.cdist(xyz, xyz)[None]
    coords, _ = distance_matrix_to_3d_structure_torch_mds(
        target,
        n_iter=1,
        device="cpu",
        progress_bar=False,
    )
    assert coords.shape == (1, n_points, 3)
    recovered = torch.cdist(torch.from_numpy(coords), torch.from_numpy(coords))
    torch.testing.assert_close(recovered, target, rtol=1e-5, atol=1e-5)


def test_smacof_is_seed_independent_and_preserves_rng_state():
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]
    )
    target = torch.cdist(xyz, xyz)[None]
    torch.manual_seed(11)
    rng_before = torch.random.get_rng_state().clone()
    first, _ = distance_matrix_to_3d_structure_torch_mds(
        target,
        n_iter=3,
        device="cpu",
        progress_bar=False,
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    torch.manual_seed(29)
    second, _ = distance_matrix_to_3d_structure_torch_mds(
        target,
        n_iter=3,
        device="cpu",
        progress_bar=False,
    )
    np.testing.assert_array_equal(first, second)


def test_smacof_batches_can_converge_at_different_iterations():
    torch.manual_seed(12)
    xyz = torch.randn(6, 3)
    maps = torch.stack([torch.zeros(6, 6), torch.cdist(xyz, xyz)])
    coords, history = distance_matrix_to_3d_structure_torch_mds(
        maps, batch_size=1, n_iter=50, device="cpu", progress_bar=False
    )
    assert coords.shape == (2, 6, 3)
    assert history.shape == (2, 50)
    assert np.isfinite(history).all()
    assert history[0, -1] == history[0, -2]


def test_weighted_smacof_preserves_bonds_on_inconsistent_maps():
    torch.manual_seed(7)
    steps = torch.randn(19, 3)
    steps = steps / steps.norm(dim=-1, keepdim=True) * 3.81
    coordinates = torch.cat((torch.zeros(1, 3), steps.cumsum(0)))
    target = torch.cdist(coordinates, coordinates)
    separation = torch.abs(torch.arange(20)[:, None] - torch.arange(20)[None, :])
    target[separation >= 8] *= 1.2
    weights = torch.ones(20, 20)
    weights[separation == 1] = 100.0
    weights.fill_diagonal_(0)

    torch.manual_seed(11)
    unweighted, _ = distance_matrix_to_3d_structure_torch_mds(
        target[None], device="cpu", progress_bar=False
    )
    torch.manual_seed(11)
    weighted, _ = distance_matrix_to_3d_structure_torch_mds(
        target[None], device="cpu", progress_bar=False, weights=weights
    )

    def bond_rmse(result):
        bond_lengths = torch.linalg.vector_norm(
            torch.diff(torch.as_tensor(result), dim=1), dim=-1
        )
        return torch.sqrt(torch.mean((bond_lengths - 3.81) ** 2))

    assert bond_rmse(weighted) < bond_rmse(unweighted)

    from starling.structure.weighted_stress import map_error_weights

    calibrated = map_error_weights(20)
    assert torch.equal(calibrated, calibrated.T)
    assert torch.count_nonzero(calibrated.diagonal()) == 0
    assert calibrated[0, 1] > calibrated[0, 8] > 0
