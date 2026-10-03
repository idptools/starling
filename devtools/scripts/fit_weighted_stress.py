"""Paired decoder-error fitting helpers for calibrate_weighted_stress.py."""

import hashlib
import json

import numpy as np

SHORT_RANGE = 6


def fit_error_model(records):
    """Fit per-separation RMS errors and a length-dependent long-range plateau."""
    if len(records) < 2:
        raise ValueError("At least two training sequences are required")
    if any(
        len(r["error_by_separation"]) != r["length"] - 1
        or not np.isfinite(r["error_by_separation"]).all()
        for r in records
    ):
        raise ValueError(
            "Each sequence needs one finite error value per residue separation"
        )
    short = np.asarray([r["error_by_separation"][:SHORT_RANGE] for r in records])
    if short.shape[1] != SHORT_RANGE:
        raise ValueError("Every sequence must have at least seven residues")
    short_error = np.sqrt(np.mean(short**2, axis=0))

    lengths = np.asarray([r["length"] for r in records], dtype=float)
    long_error = np.asarray(
        [
            np.sqrt(np.mean(np.asarray(r["error_by_separation"])[SHORT_RANGE:] ** 2))
            for r in records
        ]
    )
    if np.ptp(lengths) == 0:
        slope, intercept = 0.0, float(long_error.mean())
    else:
        slope, intercept = np.polyfit(lengths, long_error, deg=1)
    return {
        "short_separation_error": short_error.tolist(),
        "plateau_intercept": float(intercept),
        "plateau_slope": float(slope),
    }


def predict_error_profile(model, length):
    plateau = max(0.0, model["plateau_intercept"] + model["plateau_slope"] * length)
    errors = np.full(length - 1, plateau)
    errors = np.maximum(errors, 0.0)
    for i, value in enumerate(model["short_separation_error"]):
        errors[i] = min(value, plateau)
    return errors


def current_error_profile(length):
    from starling.structure.weighted_stress import (
        _ERROR_BY_SEPARATION,
        _ERROR_PLATEAU_INTERCEPT,
        _ERROR_PLATEAU_SLOPE,
    )

    plateau = _ERROR_PLATEAU_INTERCEPT + _ERROR_PLATEAU_SLOPE * length
    return np.asarray(
        [min(value, plateau) for value in _ERROR_BY_SEPARATION]
        + [plateau] * max(0, length - 7)
    )


def leave_one_family_out(records):
    """Predict each sequence with its entire family excluded from fitting."""
    if len(records) < 3:
        raise ValueError("LOO needs at least three distinct sequences")
    folds = []
    if (
        any(not r.get("family") for r in records)
        or len({r["family"] for r in records}) < 3
    ):
        raise ValueError("Family-held-out validation needs >=3 identified families")
    for held_out in records:
        training = [r for r in records if r["family"] != held_out["family"]]
        model = fit_error_model(training)
        observed = np.asarray(held_out["error_by_separation"])
        predicted = predict_error_profile(model, held_out["length"])
        baseline = current_error_profile(held_out["length"])
        folds.append(
            {
                "held_out": held_out["name"],
                "held_out_group": held_out["family"],
                "fit_sequences": [r["name"] for r in training],
                "observed_error_by_separation": observed.tolist(),
                "predicted_error_by_separation": predicted.tolist(),
                "current_error_by_separation": baseline.tolist(),
                "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
                "current_rmse": float(np.sqrt(np.mean((baseline - observed) ** 2))),
            }
        )
    return folds


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_paired_record(name, path):
    """Calibrate RMS decoder residuals from even temporal blocks only."""
    with np.load(path, allow_pickle=False) as cache:
        truth, decoded, blocks = cache["truth"], cache["decoded"], cache["blocks"]
        provenance = json.loads(cache["provenance"].item())
    if (
        truth.ndim != 3
        or truth.shape != decoded.shape
        or truth.shape[1] != truth.shape[2]
        or len(blocks) != len(truth)
        or not np.isfinite(truth).all()
        or not np.isfinite(decoded).all()
    ):
        raise ValueError(f"{name}: invalid paired distance-map cache")
    training = blocks % 2 == 0
    if not training.any():
        raise ValueError(f"{name}: no calibration frames")
    residual = decoded[training].astype(np.float64) - truth[training]
    n = truth.shape[-1]
    return {
        "name": name,
        "length": n,
        "error_by_separation": np.asarray(
            [
                np.sqrt(np.mean(np.diagonal(residual, offset=s, axis1=1, axis2=2) ** 2))
                for s in range(1, n)
            ]
        ),
        "n_calibration_frames": int(training.sum()),
        "provenance": provenance,
        "family": provenance.get("family"),
        "files": {str(path): sha256(path)},
    }
