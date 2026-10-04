import numpy as np
import pytest

from starling.structure.bme import BME
from starling.structure.bme_utils import DEFAULT_THETA, ExperimentalObservable


@pytest.mark.parametrize(
    ("constructor_theta", "expected_theta"),
    [(10.0, 10.0), (None, DEFAULT_THETA)],
)
def test_fit_without_theta_scan_uses_explicit_theta_or_default(constructor_theta, expected_theta):
    observable = ExperimentalObservable(value=1.0, uncertainty=1.0)
    calculated = np.array([[0.0], [1.0], [2.0]])
    bme = BME([observable], calculated, theta=constructor_theta)

    result = bme.fit(auto_theta=False, max_iterations=5, verbose=False)
    assert result.theta == expected_theta


@pytest.mark.parametrize("theta", [float("nan"), float("inf"), float("-inf")])
def test_constructor_rejects_nonfinite_theta(theta):
    observable = ExperimentalObservable(value=1.0, uncertainty=1.0)

    with pytest.raises(ValueError, match="theta must be finite and positive"):
        BME([observable], np.array([[0.0], [1.0], [2.0]]), theta=theta)


@pytest.mark.parametrize("theta", [float("nan"), float("inf"), float("-inf")])
def test_fit_rejects_nonfinite_theta(theta):
    observable = ExperimentalObservable(value=1.0, uncertainty=1.0)
    bme = BME([observable], np.array([[0.0], [1.0], [2.0]]))

    with pytest.raises(ValueError, match="theta must be finite and positive"):
        bme.fit(theta=theta, verbose=False)
