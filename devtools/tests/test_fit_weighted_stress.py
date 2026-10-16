import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "fit_weighted_stress.py"
SPEC = importlib.util.spec_from_file_location("fit_weighted_stress", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIT)
fit_error_model = FIT.fit_error_model
leave_one_family_out = FIT.leave_one_family_out


def test_paired_calibration_uses_training_residuals_not_ensemble_mean_bias(tmp_path):
    truth = np.zeros((8, 8, 8), dtype=np.float32)
    decoded = np.ones_like(truth)
    decoded[::2] *= np.array([1, -1, 1, -1])[:, None, None]
    decoded[1::2] *= 100  # Evaluation blocks cannot enter calibration.
    path = tmp_path / "example.paired.npz"
    np.savez(
        path,
        truth=truth,
        decoded=decoded,
        blocks=np.arange(8),
        provenance=np.asarray('{"vae_sha256": "example"}'),
    )
    record = FIT.load_paired_record("example", path)
    assert record["error_by_separation"] == pytest.approx(np.ones(7))
    assert record["n_calibration_frames"] == 4
    assert record["provenance"]["vae_sha256"] == "example"


def test_leave_one_family_out_fits_only_from_other_sequences():
    records = [
        {"name": "a", "length": 10, "error_by_separation": [1, 2, 3, 4, 5, 6, 7, 7, 7]},
        {
            "name": "b",
            "length": 20,
            "error_by_separation": [3, 4, 5, 6, 7, 8] + [9] * 13,
        },
        {
            "name": "c",
            "length": 30,
            "error_by_separation": [5, 6, 7, 8, 9, 10] + [11] * 23,
        },
    ]

    for record in records:
        record["family"] = record["name"]
    folds = leave_one_family_out(records)

    assert [fold["held_out"] for fold in folds] == ["a", "b", "c"]
    assert folds[0]["fit_sequences"] == ["b", "c"]
    assert folds[0]["predicted_error_by_separation"] == pytest.approx(
        [
            np.sqrt(17),
            np.sqrt(26),
            np.sqrt(37),
            7,
            7,
            7,
            7,
            7,
            7,
        ]
    )
    assert folds[0]["rmse"] > 0


def test_fit_error_model_recovers_length_trend():
    records = [
        {"length": 10, "error_by_separation": [1, 1, 1, 1, 1, 1, 2, 2, 2]},
        {"length": 20, "error_by_separation": [3] * 6 + [4] * 13},
    ]

    fit = fit_error_model(records)

    assert fit["short_separation_error"] == pytest.approx(
        [np.sqrt(5), np.sqrt(5), np.sqrt(5), np.sqrt(5), np.sqrt(5), np.sqrt(5)]
    )
    assert fit["plateau_intercept"] == pytest.approx(0)
    assert fit["plateau_slope"] == pytest.approx(0.2)


def test_family_grouped_validation_excludes_related_calibration_sequences():
    records = [
        {
            "name": name,
            "family": family,
            "length": n,
            "error_by_separation": [1.0] * (n - 1),
        }
        for name, family, n in [
            ("a1", "a", 10),
            ("a2", "a", 20),
            ("b", "b", 30),
            ("c", "c", 40),
        ]
    ]
    folds = FIT.leave_one_family_out(records)
    assert folds[0]["fit_sequences"] == ["b", "c"]
    assert folds[1]["fit_sequences"] == ["b", "c"]
    assert folds[0]["held_out_group"] == "a"


@pytest.mark.parametrize("too_short", [5, 7])
def test_fit_error_model_rejects_sequences_without_long_range_separations(too_short):
    """A 7-residue sequence has separations 1-6 only, so its long-range slice is empty
    and the plateau fit would silently become NaN; it must be rejected instead."""
    records = [
        {"length": too_short, "error_by_separation": [1.0] * (too_short - 1)},
        {"length": 20, "error_by_separation": [3] * 6 + [4] * 13},
    ]

    with pytest.raises(ValueError, match="at least 8 residues"):
        fit_error_model(records)


def test_fit_error_model_accepts_shortest_sequence_with_a_long_range_separation():
    """An 8-residue sequence has one long-range separation, which is enough to fit."""
    records = [
        {"length": FIT.MIN_RESIDUES, "error_by_separation": [1.0] * 6 + [2.0]},
        {"length": 20, "error_by_separation": [3] * 6 + [4] * 13},
    ]

    fit = fit_error_model(records)

    assert np.isfinite(fit["plateau_intercept"])
    assert np.isfinite(fit["plateau_slope"])
