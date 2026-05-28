"""MSD calculations and species grouping."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np

from .io import Source, load_frames
from .unwrap import unwrap_positions


def self_msd(
    positions: np.ndarray,
    max_lag: int | None = None,
    atom_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Compute the scalar self mean square displacement for a position trajectory."""
    coordinates = np.asarray(positions, dtype=float)
    if coordinates.ndim != 3 or coordinates.shape[2] != 3:
        raise ValueError("positions must have shape (n_frames, n_atoms, 3).")
    if coordinates.shape[0] < 2:
        raise ValueError("positions must contain at least two frames.")

    lag_limit = _normalize_max_lag(coordinates.shape[0], max_lag)
    selected = _select_atoms(coordinates, atom_indices)

    msd = np.zeros(lag_limit + 1, dtype=float)
    for lag in range(1, lag_limit + 1):
        displacement = selected[lag:] - selected[:-lag]
        squared_displacement = np.sum(displacement * displacement, axis=-1)
        msd[lag] = float(np.mean(squared_displacement))

    return msd


def msd_by_species_from_frames(
    frames: Source,
    max_lag: int | None = None,
    timestep: Real | None = None,
) -> dict[str, object]:
    """Compute overall and per-symbol self MSD from frames or an ASE trajectory source."""
    normalized_frames = load_frames(frames)
    positions = unwrap_positions(normalized_frames)

    first_frame_symbols = normalized_frames[0].get_chemical_symbols()
    symbols = tuple(dict.fromkeys(first_frame_symbols))
    lagged_overall = self_msd(positions, max_lag=max_lag)
    lag_frames = np.arange(lagged_overall.shape[0], dtype=int)
    lag_times = None if timestep is None else lag_frames * float(timestep)

    by_symbol = {}
    for symbol in symbols:
        atom_indices = [
            index for index, value in enumerate(first_frame_symbols) if value == symbol
        ]
        by_symbol[symbol] = self_msd(
            positions,
            max_lag=max_lag,
            atom_indices=atom_indices,
        )

    return {
        "lag_frames": lag_frames,
        "lag_times": lag_times,
        "overall": lagged_overall,
        "by_symbol": by_symbol,
        "symbols": symbols,
    }


def msd_by_species_from_file(
    path: str,
    max_lag: int | None = None,
    timestep: Real | None = None,
) -> dict[str, object]:
    """Load a trajectory file and compute species-resolved self MSD."""
    return msd_by_species_from_frames(load_frames(path), max_lag=max_lag, timestep=timestep)


def _normalize_max_lag(n_frames: int, max_lag: int | None) -> int:
    if max_lag is None:
        return n_frames - 1
    if not isinstance(max_lag, Integral):
        raise ValueError("max_lag must be an integer.")
    if max_lag < 0 or max_lag >= n_frames:
        raise ValueError("max_lag must be between 0 and n_frames - 1.")
    return int(max_lag)


def _select_atoms(positions: np.ndarray, atom_indices: Sequence[int] | None) -> np.ndarray:
    if atom_indices is None:
        return positions

    indices = np.asarray(atom_indices, dtype=int)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("atom_indices must be a non-empty one-dimensional sequence.")
    if np.any(indices < 0) or np.any(indices >= positions.shape[1]):
        raise ValueError("atom_indices must refer to valid atom positions.")

    return positions[:, indices, :]
