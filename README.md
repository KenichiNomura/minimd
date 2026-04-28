# minimd

A minimal molecular dynamics driver using machine-learning interatomic potentials (MLIPs) via the ASE framework.

## Overview

`minimd.py` runs a two-stage simulation on an input atomic structure:

1. **Geometry relaxation** — FIRE optimizer minimizes forces until convergence (`fmax`)
2. **NVT thermalization** — Langevin dynamics at a target temperature for a set number of steps

Trajectories are saved as both `.traj` (ASE binary) and `.xyz` files under a subdirectory named after the chosen model.

## Supported models

| Flag | Model |
|------|-------|
| `mace` | [MACE-MP](https://github.com/ACEsuit/mace) — MACE reference implementation (medium, float64) |
| `nequip` | [NequIP](https://github.com/mir-group/nequip) — E(3)-equivariant potentials; pre-trained models at https://www.nequip.net |
| `allegro` | [Allegro](https://github.com/mir-group/allegro) — scalable equivariant interatomic potentials |
| `mattersim` | MatterSim — project page / repository (TBD) |
| `sevennet` | SevenNet — project page / repository (TBD) |
| `upet` | UPET — project page / repository (TBD) |
| `uma` | UMA via FAIRChem — project page / repository (TBD) |

## Usage

```
python minimd.py [-c CONFIG] [-m MODEL] [-f FMAX] [-s STEPS] [-t TEMPERATURE]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-c`, `--config` | `sio2.xyz` | Input structure file (any ASE-readable format; `.lammpstrj` auto-detected as LAMMPS data) |
| `-m`, `--model` | `mace` | MLIP backend (see table above) |
| `-f`, `--fmax` | `2.0` | Force convergence threshold for relaxation (eV/Å) |
| `-s`, `--steps` | `1000` | Number of MD steps |
| `-t`, `--temperature` | `300.0` | Target temperature for Langevin thermostat (K) |

### Example

```bash
python minimd.py -c sio2.xyz -m mace -f 0.05 -s 5000 -t 1000
```

## Output

All output is written to `<model>/`:

| File | Contents |
|------|----------|
| `minimd.log` | General log |
| `md.log` | Per-step MD energies/temperatures |
| `relax.traj` / `relax.xyz` | Relaxation trajectory |
| `md.traj` / `md.xyz` | MD trajectory |

## Dependencies

- Python >= 3.11
- [ASE](https://wiki.fysik.dtu.dk/ase/) >= 3.28
- [PyTorch](https://pytorch.org/) >= 2.11
- [MACE-torch](https://github.com/ACEsuit/mace) >= 0.3.15 (required for `mace` model; others are optional)

GPU acceleration is used automatically when CUDA is available.

## Environment (using uv)

This project uses a uv lockfile (uv.lock) to pin Python dependencies. Recommended steps to create a reproducible environment:

1. Ensure Python 3.11+ and pip are installed.
2. Install the uv CLI if not present:

   pip install uv

3. Sync the environment from the lockfile (creates and installs into a virtual environment):

   uv sync

4. Activate the created venv (if uv places it at .venv):

   - macOS / Linux: source .venv/bin/activate
   - Windows (PowerShell): .\.venv\Scripts\Activate.ps1

If uv creates the environment at another path, follow uv's output to activate it. For GPU support, install the appropriate CUDA build of PyTorch following https://pytorch.org/ after syncing.

## References

- MACE: ACEsuit MACE repository — https://github.com/ACEsuit/mace. Documentation and citations: https://mace-docs.readthedocs.io/ and the paper linked from the repository (see "References").

- NequIP: NequIP repository — https://github.com/mir-group/nequip. Docs and citation info: https://nequip.readthedocs.io/ (see "References & citing" in the repo).

- Allegro: Allegro repository — https://github.com/mir-group/allegro (see repository README for citation details).

- MatterSim, SevenNet, UPET, UMA: project pages / canonical repositories and publication references are not unambiguously discoverable from this repository. Please provide preferred links or allow me to look them up and add exact citations.

(If preferred, exact paper citations (authors, title, venue, year, DOI) can be added to each entry.)


