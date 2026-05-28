"""Trajectory loading and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.io import read


@dataclass(frozen=True)
class TrajectoryMetadata:
    symbols: tuple[str, ...]
    cell: np.ndarray
    pbc: tuple[bool, bool, bool]


Source = str | PathLike[str] | Path | Iterable[Atoms]


def load_frames(source: Source) -> list[Atoms]:
    """Load a trajectory source and validate the resulting frame sequence."""
    frames = _coerce_frames(source)
    validate_frames(frames)
    return frames


def validate_frames(frames: list[Atoms]) -> TrajectoryMetadata:
    """Validate that a trajectory matches the fixed-cell assumptions for v1."""
    if len(frames) < 2:
        raise ValueError("Trajectory must contain at least two frames.")

    first = frames[0]
    symbols = tuple(first.get_chemical_symbols())
    cell = np.asarray(first.cell.array, dtype=float)
    pbc = tuple(bool(value) for value in np.asarray(first.pbc, dtype=bool))

    if cell.shape != (3, 3) or np.linalg.matrix_rank(cell) < 3:
        raise ValueError("Trajectory must define a full-rank 3x3 simulation cell.")

    atom_count = len(first)
    for frame in frames[1:]:
        if len(frame) != atom_count:
            raise ValueError("Trajectory must keep a constant atom count across frames.")
        if tuple(frame.get_chemical_symbols()) != symbols:
            raise ValueError("Trajectory must keep the atom order and symbols fixed across frames.")
        if tuple(bool(value) for value in np.asarray(frame.pbc, dtype=bool)) != pbc:
            raise ValueError("Trajectory must keep PBC flags fixed across frames.")
        if not np.allclose(np.asarray(frame.cell.array, dtype=float), cell):
            raise ValueError("Trajectory must keep the simulation cell fixed across frames.")

    return TrajectoryMetadata(symbols=symbols, cell=cell, pbc=pbc)


def _coerce_frames(source: Source) -> list[Atoms]:
    if isinstance(source, (str, PathLike, Path)):
        frames = read(str(source), index=":")
    else:
        frames = list(source)

    if isinstance(frames, Atoms):
        frames = [frames]

    normalized: list[Atoms] = []
    for frame in frames:
        if not isinstance(frame, Atoms):
            raise TypeError("Trajectory inputs must be ASE Atoms objects.")
        normalized.append(frame.copy())

    return normalized
