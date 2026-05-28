from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.constraints import FixCom
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.spacegroup import crystal

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "generated"


def build_fcc_aluminum() -> Atoms:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 2))
    atoms.calc = EMT()
    atoms.set_constraint(FixCom())
    return atoms


def build_alpha_quartz() -> Atoms:
    atoms = crystal(
        ["Si", "O"],
        basis=[(0.4697, 0.0, 0.0), (0.4133, 0.2672, 0.1188)],
        spacegroup=152,
        cellpar=[4.913, 4.913, 5.405, 90.0, 90.0, 120.0],
    ).repeat((2, 2, 2))
    atoms.calc = LennardJones(epsilon=0.0103, sigma=1.4, rc=4.0)
    atoms.set_constraint(FixCom())
    return atoms


def initialize_temperature(atoms: Atoms, temperature_k: float, seed: int) -> None:
    rng = np.random.RandomState(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=rng)
    Stationary(atoms)


def snapshot(atoms: Atoms, step: int) -> Atoms:
    frame = atoms.copy()
    frame.info["step"] = step
    frame.info["temperature_K"] = float(atoms.get_temperature())
    return frame


def run_md(
    atoms: Atoms,
    *,
    temperature_k: float,
    timestep_fs: float,
    steps: int,
    sample_interval: int,
    friction: float,
    seed: int,
) -> list[Atoms]:
    initialize_temperature(atoms, temperature_k=temperature_k, seed=seed)
    dynamics = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_k,
        friction=friction,
        fixcm=False,
        rng=np.random.RandomState(seed + 1),
    )

    frames = [snapshot(atoms, step=0)]
    for step in range(1, steps + 1):
        dynamics.run(1)
        atoms.wrap()
        if step % sample_interval == 0 or step == steps:
            frames.append(snapshot(atoms, step=step))
    return frames


def write_trajectory(path: Path, frames: list[Atoms]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, frames, format="extxyz")


def main() -> None:
    aluminum_frames = run_md(
        build_fcc_aluminum(),
        temperature_k=300.0,
        timestep_fs=2.0,
        steps=10000,
        sample_interval=5,
        friction=0.01,
        seed=7,
    )
    quartz_frames = run_md(
        build_alpha_quartz(),
        temperature_k=300.0,
        timestep_fs=1.0,
        steps=10000,
        sample_interval=5,
        friction=0.02,
        seed=11,
    )

    aluminum_path = OUTPUT_DIR / "fcc_al_300k.extxyz"
    quartz_path = OUTPUT_DIR / "quartz_sio2_300k.extxyz"
    write_trajectory(aluminum_path, aluminum_frames)
    write_trajectory(quartz_path, quartz_frames)

    print(
        f"Wrote {aluminum_path} with {len(aluminum_frames)} frames "
        f"and {len(aluminum_frames[0])} atoms"
    )
    print(
        f"Wrote {quartz_path} with {len(quartz_frames)} frames "
        f"and {len(quartz_frames[0])} atoms"
    )


if __name__ == "__main__":
    main()
