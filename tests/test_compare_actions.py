import numpy as np

from tools.compare_actions import denormalize, from_relative, normalize, to_relative


def test_normalization_round_trip():
    values = np.array([[1.0, 2.0]], dtype=np.float64)
    stats = {
        "mean": np.array([0.5, 1.5]),
        "std": np.array([0.5, 0.25]),
    }

    assert np.allclose(denormalize(normalize(values, stats), stats), values)


def test_relative_conversion_happens_once():
    state = np.array([[1.0, 3.0]])
    target = np.array([[1.5, 2.0]])

    assert np.allclose(from_relative(to_relative(target, state), state), target)
