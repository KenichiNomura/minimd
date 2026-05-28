# MSDwork

## Purpose

`msdwork` is a library-first Python project for computing **species-resolved self mean square displacement (MSD)** from **wrapped ASE trajectories**.

The current v1 target is:

- input trajectories primarily in **extended XYZ (`.extxyz`)** format
- **fixed-cell** periodic systems only
- support for **triclinic** as well as orthorhombic cells
- return MSD **per chemical symbol** plus an **overall** MSD curve
- return lag in **frames by default**, with optional physical lag times when an explicit `timestep` is passed

## Scope

### In Scope

- loading ASE trajectories from file or in-memory `Atoms` frames
- fixed-cell trajectory validation
- wrapped-to-unwrapped coordinate reconstruction under PBC
- scalar self-MSD calculation
- species-resolved MSD grouped by ASE chemical symbol
- automated tests and CI for the analysis workflow

### Out of Scope for v1

- variable-cell or NPT trajectory support
- CLI tooling
- tensorial MSD
- diffusion coefficient fitting
- distinct/pair MSD variants
- production-quality silica force fields
- automatic timestep extraction from file metadata

## Development Environment

This project is developed in:

- Python `3.12`
- `uv` for environment and dependency management

Standard local commands:

```bash
uv sync
uv run pytest
uv run ruff check .
```

## Public API

### `load_frames(source) -> list[ase.Atoms]`

Accepts either:

- a path to a trajectory file, typically `.extxyz`
- an iterable of ASE `Atoms` frames

Returns a validated list of copied `Atoms` frames.

### `unwrap_positions(frames) -> np.ndarray`

Returns continuous Cartesian coordinates with shape:

```python
(n_frames, n_atoms, 3)
```

This function reconstructs unwrapped positions from wrapped coordinates using fractional-coordinate nearest-image correction.

### `self_msd(positions, max_lag=None, atom_indices=None) -> np.ndarray`

Computes scalar self-MSD values for lags:

```python
0 .. max_lag
```

If `atom_indices` is provided, the MSD is averaged only over those atoms.

### `msd_by_species_from_frames(frames, max_lag=None, timestep=None) -> dict`

High-level species-resolved analysis from an in-memory trajectory.

### `msd_by_species_from_file(path, max_lag=None, timestep=None) -> dict`

High-level species-resolved analysis from a trajectory file.

## Result Schema

The high-level analysis functions return a dict with:

- `lag_frames`: 1D integer array of lag indices
- `lag_times`: `None` if `timestep is None`, otherwise `lag_frames * timestep`
- `overall`: 1D MSD array averaged over all atoms
- `by_symbol`: dict mapping symbols like `H`, `O`, `Si`, `Al` to 1D MSD arrays
- `symbols`: tuple of unique symbols in first-appearance order

## Scientific and Data Assumptions

### Atom Type Definition

“Atom type” means the **ASE chemical symbol**.

### Trajectory Assumptions

All frames must preserve:

- constant atom count
- constant atom ordering
- constant chemical symbols
- constant simulation cell
- constant PBC flags

### Cell Assumptions

The cell must be a valid full-rank `3x3` cell matrix.

### Time Axis Assumptions

Physical time is **not inferred** automatically from `.extxyz` metadata in v1.
If physical lag times are needed, pass `timestep` explicitly.

## Code Design

### Package Layout

- `src/msdwork/__init__.py`
  - public exports only
- `src/msdwork/io.py`
  - loading and trajectory-level validation
- `src/msdwork/unwrap.py`
  - wrapped-to-unwrapped coordinate reconstruction
- `src/msdwork/msd.py`
  - MSD math and species grouping
- `scripts/generate_sample_trajectories.py`
  - ASE-based example trajectory generation

### Loading and Validation Design

`load_frames(...)` normalizes the input into a list of copied ASE frames and validates the trajectory before analysis.

Validation checks:

- at least 2 frames
- constant atom count across frames
- constant symbol ordering across frames
- constant PBC flags across frames
- constant cell across frames
- full-rank `3x3` cell

These constraints are intentional for v1 to keep the unwrapping and MSD logic well-defined.

### Unwrapping Design

The unwrapping algorithm works in **fractional coordinates**:

1. use the first frame as the structural reference
2. convert each frame to fractional coordinates with `wrap=False`
3. compute frame-to-frame fractional displacements
4. on periodic axes, apply nearest-image correction via:

```python
delta -= round(delta)
```

5. on non-periodic axes, keep the raw displacement
6. accumulate corrected fractional displacements over time
7. convert accumulated fractional coordinates back to Cartesian coordinates

This is the key design choice that makes triclinic-cell support straightforward.

### MSD Design

`self_msd(...)` expects an array of shape:

```python
(n_frames, n_atoms, 3)
```

For each lag `tau`:

1. compute displacements between frames `t` and `t + tau`
2. square and sum over `x, y, z`
3. average over valid time origins
4. average over selected atoms or all atoms

The result is one scalar MSD value per lag.

### Species Grouping Design

Species grouping is based on the first frame’s chemical symbols.

Behavior:

- unique symbols are collected in first-appearance order
- one atom-index group is built per symbol
- one `self_msd(...)` call is run for the full system
- one `self_msd(...)` call is run for each symbol group
- results are assembled into the shared result schema

## Test Design

The project uses `pytest` with three main test modules:

- `tests/test_self_msd.py`
- `tests/test_unwrap_positions.py`
- `tests/test_species_msd_integration.py`

### MSD Unit Tests

- lag 0 returns zero
- single-atom known displacement case
- averaging over time origins
- averaging over selected atoms
- squared displacement summed over dimensions
- full curve through `max_lag`
- invalid `max_lag` is rejected

### Unwrapping Unit Tests

- single orthorhombic boundary crossing
- repeated crossings
- non-periodic axis remains unwrapped
- triclinic crossing reconstructed correctly in fractional space
- missing cell/PBC data is rejected
- changing cell/PBC is rejected in v1

### Species and Integration Tests

- mixed symbols return separate curves and overall MSD
- symbol order is deterministic
- single-atom species work correctly
- `.extxyz` round-trip works
- file-based API matches frame-based API
- short ASE MD smoke test returns finite, non-negative curves

## CI/CD

GitHub Actions is the current CI platform.

Workflow file:

- `.github/workflows/ci.yml`

Current CI steps:

1. set up Python 3.12
2. install `uv`
3. run `uv sync`
4. run `uv run ruff check .`
5. run `uv run pytest`

Current scope is CI only. Automated release packaging can be added later.

## TDD Backbone

The project is meant to be developed in a Kent Beck–style TDD rhythm:

1. write the smallest failing test
2. make it pass with the smallest change
3. refactor only when green
4. keep the tests concrete and hand-checkable when possible

Preferred order for future feature work:

1. pure MSD math
2. trajectory unwrapping
3. species grouping behavior
4. file I/O and end-to-end workflow

## Example Trajectories

Use the helper script to generate two sample trajectories:

```bash
uv run python scripts/generate_sample_trajectories.py
```

Generated outputs:

- `generated/fcc_al_300k.extxyz`
- `generated/quartz_sio2_300k.extxyz`

### Notes on the Sample Systems

- **FCC aluminum** uses ASE’s `EMT` calculator.
- **Quartz SiO2** is generated with ASE structure tools and propagated with an ASE built-in `LennardJones` calculator as a lightweight demonstrator.

The quartz sample is useful for workflow and testing, but it should **not** be treated as a physically reliable silica MD model.

## Current Usage Examples

### High-Level File API

```python
from msdwork import msd_by_species_from_file

result = msd_by_species_from_file("generated/quartz_sio2_300k.extxyz", timestep=1.0)

print(result["symbols"])
print(result["lag_frames"])
print(result["lag_times"])
print(result["overall"])
print(result["by_symbol"]["Si"])
print(result["by_symbol"]["O"])
```

### Low-Level API

```python
from msdwork import load_frames, self_msd, unwrap_positions

frames = load_frames("generated/fcc_al_300k.extxyz")
positions = unwrap_positions(frames)
overall_msd = self_msd(positions)
```

## Future Extensions

Good next candidates after v1 stability:

- variable-cell trajectory support
- CLI entry point
- timestep metadata conventions for `.extxyz`
- tensorial MSD
- diffusion fitting utilities
- plotting helpers
- more realistic SiO2 force fields or external-engine workflows
