import numpy as np
import pytest
from ase import Atoms

from msdwork import unwrap_positions


def make_frame(symbols, scaled_positions, cell, pbc):
    frame = Atoms(symbols=symbols, cell=cell, pbc=pbc)
    frame.set_scaled_positions(np.array(scaled_positions, dtype=float))
    return frame


def test_unwraps_single_crossing_in_orthorhombic_cell():
    cell = np.diag([10.0, 10.0, 10.0])
    frames = [
        make_frame(["H"], [[0.9, 0.2, 0.3]], cell, [True, True, True]),
        make_frame(["H"], [[0.1, 0.2, 0.3]], cell, [True, True, True]),
    ]

    positions = unwrap_positions(frames)

    expected = np.array(
        [
            [[9.0, 2.0, 3.0]],
            [[11.0, 2.0, 3.0]],
        ]
    )
    np.testing.assert_allclose(positions, expected)


def test_unwraps_repeated_crossings():
    cell = np.diag([10.0, 10.0, 10.0])
    frames = [
        make_frame(["H"], [[0.9, 0.2, 0.3]], cell, [True, True, True]),
        make_frame(["H"], [[0.1, 0.2, 0.3]], cell, [True, True, True]),
        make_frame(["H"], [[0.3, 0.2, 0.3]], cell, [True, True, True]),
    ]

    positions = unwrap_positions(frames)

    expected = np.array(
        [
            [[9.0, 2.0, 3.0]],
            [[11.0, 2.0, 3.0]],
            [[13.0, 2.0, 3.0]],
        ]
    )
    np.testing.assert_allclose(positions, expected)


def test_nonperiodic_axis_is_not_wrapped():
    cell = np.diag([10.0, 10.0, 10.0])
    frames = [
        make_frame(["H"], [[0.2, 0.9, 0.1]], cell, [True, False, False]),
        make_frame(["H"], [[0.2, 0.1, 0.1]], cell, [True, False, False]),
    ]

    positions = unwrap_positions(frames)

    expected = np.array(
        [
            [[2.0, 9.0, 1.0]],
            [[2.0, 1.0, 1.0]],
        ]
    )
    np.testing.assert_allclose(positions, expected)


def test_unwraps_triclinic_crossing_using_fractional_coordinates():
    cell = np.array(
        [
            [10.0, 0.0, 0.0],
            [2.0, 8.0, 0.0],
            [1.0, 1.0, 6.0],
        ]
    )
    scaled_positions = [
        np.array([[0.9, 0.1, 0.2]]),
        np.array([[0.1, 0.1, 0.2]]),
    ]
    frames = [
        make_frame(["H"], scaled_positions[0], cell, [True, True, True]),
        make_frame(["H"], scaled_positions[1], cell, [True, True, True]),
    ]

    positions = unwrap_positions(frames)

    first = scaled_positions[0] @ cell
    expected = np.stack([first, first + np.array([[0.2, 0.0, 0.0]]) @ cell])
    np.testing.assert_allclose(positions, expected)


def test_rejects_missing_cell_or_pbc_data():
    bad_frames = [
        Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.zeros((3, 3)), pbc=[True, True, True]),
        Atoms("H", positions=[[1.0, 0.0, 0.0]], cell=np.zeros((3, 3)), pbc=[True, True, True]),
    ]

    with pytest.raises(ValueError):
        unwrap_positions(bad_frames)


def test_rejects_changing_cell_or_pbc_in_v1():
    frames = [
        make_frame(["H"], [[0.1, 0.2, 0.3]], np.diag([10.0, 10.0, 10.0]), [True, True, True]),
        make_frame(["H"], [[0.2, 0.2, 0.3]], np.diag([12.0, 10.0, 10.0]), [True, True, True]),
    ]

    with pytest.raises(ValueError):
        unwrap_positions(frames)

    frames = [
        make_frame(["H"], [[0.1, 0.2, 0.3]], np.diag([10.0, 10.0, 10.0]), [True, True, True]),
        make_frame(["H"], [[0.2, 0.2, 0.3]], np.diag([10.0, 10.0, 10.0]), [True, False, True]),
    ]

    with pytest.raises(ValueError):
        unwrap_positions(frames)
