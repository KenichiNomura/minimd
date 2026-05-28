import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write
from ase.md.verlet import VelocityVerlet

from msdwork import load_frames, msd_by_species_from_file, msd_by_species_from_frames


def make_frame(symbols, scaled_positions, cell, pbc):
    frame = Atoms(symbols=symbols, cell=cell, pbc=pbc)
    frame.set_scaled_positions(np.array(scaled_positions, dtype=float))
    return frame


def make_species_test_frames():
    cell = np.diag([10.0, 10.0, 10.0])
    symbols = ["H", "H", "O", "Si"]
    scaled_frames = [
        [
            [0.9, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.1],
            [0.3, 0.3, 0.3],
        ],
        [
            [0.0, 0.1, 0.1],
            [0.2, 0.1, 0.1],
            [0.2, 0.4, 0.1],
            [0.3, 0.3, 0.3],
        ],
        [
            [0.1, 0.1, 0.1],
            [0.3, 0.1, 0.1],
            [0.2, 0.6, 0.1],
            [0.3, 0.3, 0.3],
        ],
    ]
    return [make_frame(symbols, coords, cell, [True, True, True]) for coords in scaled_frames]


def test_groups_msd_by_symbol_and_returns_overall():
    result = msd_by_species_from_frames(make_species_test_frames())

    assert result["symbols"] == ("H", "O", "Si")
    assert result["lag_times"] is None
    np.testing.assert_array_equal(result["lag_frames"], np.array([0, 1, 2]))
    np.testing.assert_allclose(result["overall"], np.array([0.0, 1.5, 6.0]))
    np.testing.assert_allclose(result["by_symbol"]["H"], np.array([0.0, 1.0, 4.0]))
    np.testing.assert_allclose(result["by_symbol"]["O"], np.array([0.0, 4.0, 16.0]))
    np.testing.assert_allclose(result["by_symbol"]["Si"], np.array([0.0, 0.0, 0.0]))


def test_symbols_are_reported_in_deterministic_first_appearance_order():
    result = msd_by_species_from_frames(make_species_test_frames())

    assert result["symbols"] == ("H", "O", "Si")
    assert tuple(result["by_symbol"]) == ("H", "O", "Si")


def test_single_atom_species_is_supported():
    result = msd_by_species_from_frames(make_species_test_frames())

    np.testing.assert_allclose(result["by_symbol"]["O"], np.array([0.0, 4.0, 16.0]))
    np.testing.assert_allclose(result["by_symbol"]["Si"], np.array([0.0, 0.0, 0.0]))


def test_loads_short_extxyz_and_computes_species_msd(tmp_path):
    path = tmp_path / "species.extxyz"
    frames = make_species_test_frames()
    write(path, frames, format="extxyz")

    loaded_frames = load_frames(path)
    result = msd_by_species_from_file(path)

    assert len(loaded_frames) == len(frames)
    np.testing.assert_allclose(result["by_symbol"]["H"], np.array([0.0, 1.0, 4.0]))


def test_end_to_end_file_api_matches_frames_api(tmp_path):
    path = tmp_path / "species.extxyz"
    frames = make_species_test_frames()
    write(path, frames, format="extxyz")

    from_frames = msd_by_species_from_frames(frames, timestep=0.5)
    from_file = msd_by_species_from_file(path, timestep=0.5)

    assert from_file["symbols"] == from_frames["symbols"]
    np.testing.assert_array_equal(from_file["lag_frames"], from_frames["lag_frames"])
    np.testing.assert_allclose(from_file["lag_times"], np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(from_file["lag_times"], from_frames["lag_times"])
    np.testing.assert_allclose(from_file["overall"], from_frames["overall"])
    for symbol in from_frames["symbols"]:
        np.testing.assert_allclose(from_file["by_symbol"][symbol], from_frames["by_symbol"][symbol])


@pytest.mark.integration
def test_fixed_seed_ase_md_smoke_test_returns_nonnegative_curves(tmp_path):
    rng = np.random.default_rng(0)
    atoms = bulk("Cu", cubic=True)
    atoms.calc = EMT()
    atoms.set_velocities(rng.normal(scale=0.01, size=(len(atoms), 3)))

    dynamics = VelocityVerlet(atoms, timestep=1.0 * units.fs)
    frames = [atoms.copy()]
    for _ in range(3):
        dynamics.run(1)
        atoms.wrap()
        frames.append(atoms.copy())

    path = tmp_path / "cu_md.extxyz"
    write(path, frames, format="extxyz")

    result = msd_by_species_from_file(path)

    assert result["symbols"] == ("Cu",)
    assert np.all(np.isfinite(result["overall"]))
    assert np.all(result["overall"] >= 0.0)
    assert np.all(np.isfinite(result["by_symbol"]["Cu"]))
    assert np.all(result["by_symbol"]["Cu"] >= 0.0)
