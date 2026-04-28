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
| `mace` | MACE-MP (medium, float64) |
| `nequip` | NequIP — requires `NequIP-OAM-L-0.1.nequip.pth` |
| `allegro` | Allegro — requires `Allegro-OAM-L-0.1.nequip.pth` |
| `mattersim` | MatterSim |
| `sevennet` | SevenNet (7net-omni) |
| `upet` | UPET (pet-omat-m v1.0.0) |
| `uma` | UMA via FAIRChem — requires `uma-s-1p1.pt` |

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
