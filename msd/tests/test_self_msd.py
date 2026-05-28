import numpy as np
import pytest

from msdwork import self_msd


def test_lag_zero_is_zero():
    positions = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
        ]
    )

    result = self_msd(positions)

    np.testing.assert_allclose(result, np.array([0.0, 1.0]))


def test_single_atom_single_dimension_known_lag_one():
    positions = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
        ]
    )

    result = self_msd(positions, max_lag=1)

    np.testing.assert_allclose(result, np.array([0.0, 4.0]))


def test_averages_over_time_origins():
    positions = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0]],
        ]
    )

    result = self_msd(positions, max_lag=1)

    np.testing.assert_allclose(result, np.array([0.0, 2.5]))


def test_averages_over_selected_atoms():
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ]
    )

    result = self_msd(positions, max_lag=1, atom_indices=[1])

    np.testing.assert_allclose(result, np.array([0.0, 4.0]))


def test_sums_squared_displacement_over_xyz():
    positions = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 2.0, 2.0]],
        ]
    )

    result = self_msd(positions, max_lag=1)

    np.testing.assert_allclose(result, np.array([0.0, 9.0]))


def test_returns_curve_from_zero_to_max_lag():
    positions = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0]],
        ]
    )

    result = self_msd(positions, max_lag=2)

    np.testing.assert_allclose(result, np.array([0.0, 1.0, 4.0]))


@pytest.mark.parametrize("max_lag", [-1, 2.5, 4])
def test_rejects_invalid_max_lag(max_lag):
    positions = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
        ]
    )

    with pytest.raises(ValueError):
        self_msd(positions, max_lag=max_lag)
