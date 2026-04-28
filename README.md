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
| `mace` | [MACE-MP](https://github.com/ACEsuit/mace) — MACE reference implementation (medium, float64). Paper: NeurIPS 2022 (OpenReview) https://openreview.net/forum?id=YPpSngE-ZU. Code DOI: https://doi.org/10.5281/zenodo.14103332 |
| `nequip` | [NequIP](https://github.com/mir-group/nequip) — E(3)-equivariant potentials; pre-trained models at https://www.nequip.net. Paper: Nature Communications 2022, DOI: https://doi.org/10.1038/s41467-022-29939-5. Code DOI: https://doi.org/10.5281/zenodo.18200066 |
| `allegro` | [Allegro](https://github.com/mir-group/allegro) — scalable equivariant interatomic potentials. Paper: Nature Communications 2023, DOI: https://doi.org/10.1038/s41467-023-36329-y |
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

- MACE
  - Repository: https://github.com/ACEsuit/mace
  - Paper (NeurIPS 2022): https://openreview.net/forum?id=YPpSngE-ZU
  - Code / release DOI: https://doi.org/10.5281/zenodo.14103332

- NequIP
  - Repository: https://github.com/mir-group/nequip
  - Paper (Nature Communications 2022): https://doi.org/10.1038/s41467-022-29939-5
  - Code / release DOI: https://doi.org/10.5281/zenodo.18200066
  - Docs: https://nequip.readthedocs.io/

- Allegro
  - Repository: https://github.com/mir-group/allegro
  - Paper (Nature Communications 2023): https://doi.org/10.1038/s41467-023-36329-y
  - Docs: https://allegro.readthedocs.io/

- MatterSim, SevenNet, UPET, UMA
  - Canonical project pages and publication DOIs could not be unambiguously identified automatically. If you provide the exact URLs or DOIs for these models I will add them; otherwise, permission to continue searching is requested.

(If preferred, entries can be expanded to full BibTeX references.)


