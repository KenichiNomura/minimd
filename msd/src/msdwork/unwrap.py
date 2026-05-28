"""Coordinate unwrapping for wrapped ASE trajectories."""

from __future__ import annotations

import numpy as np

from .io import Source, load_frames


def unwrap_positions(frames: Source) -> np.ndarray:
    """Reconstruct continuous Cartesian coordinates from wrapped trajectory frames."""
    normalized_frames = load_frames(frames)
    cell = np.asarray(normalized_frames[0].cell.array, dtype=float)
    pbc = np.asarray(normalized_frames[0].pbc, dtype=bool)

    scaled_positions = np.stack(
        [frame.get_scaled_positions(wrap=False) for frame in normalized_frames],
        axis=0,
    )
    unwrapped_scaled = np.empty_like(scaled_positions)
    unwrapped_scaled[0] = scaled_positions[0]

    for frame_index in range(1, len(normalized_frames)):
        delta = scaled_positions[frame_index] - scaled_positions[frame_index - 1]
        delta[:, pbc] -= np.round(delta[:, pbc])
        unwrapped_scaled[frame_index] = unwrapped_scaled[frame_index - 1] + delta

    return unwrapped_scaled @ cell
