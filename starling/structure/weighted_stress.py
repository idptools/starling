"""Per-pair weights for generated distance maps."""

import torch

# Posterior-sample calibration: devtools/scripts/weighted_stress_expanded_study.json.
_ERROR_BY_SEPARATION = (
    0.1780475402817346,
    0.6201276379083143,
    0.8913920138032031,
    1.0280094843206873,
    1.0810112638333218,
    1.1067225738830264,
)
_ERROR_PLATEAU_INTERCEPT = 0.7130944426723491
_ERROR_PLATEAU_SLOPE = 0.002520151630153439


def map_error_weights(n, device="cpu"):
    """Return inverse squared map-error weights indexed by residue separation."""
    index = (
        torch.arange(n, device=device)[:, None] - torch.arange(n, device=device)
    ).abs()
    plateau = _ERROR_PLATEAU_INTERCEPT + _ERROR_PLATEAU_SLOPE * n
    error = torch.full((n, n), plateau, device=device)
    for separation, value in enumerate(_ERROR_BY_SEPARATION, start=1):
        error[index == separation] = min(value, plateau)
    weights = error.square().reciprocal()
    weights.fill_diagonal_(0)
    return weights
