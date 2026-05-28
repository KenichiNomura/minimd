from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from msdwork import msd_by_species_from_file

ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    ("fcc_al_300k.extxyz", 2.0, "FCC Aluminum at 300 K"),
    ("quartz_sio2_300k.extxyz", 1.0, "Quartz SiO2 at 300 K"),
]


def plot_trajectory(filename: str, timestep_fs: float, title: str) -> Path:
    trajectory_path = ROOT / "generated" / filename
    result = msd_by_species_from_file(str(trajectory_path), timestep=timestep_fs)

    lag_times = np.asarray(result["lag_times"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    for symbol in result["symbols"]:
        ax.plot(lag_times, result["by_symbol"][symbol], linewidth=2, label=symbol)

    ax.set_title(title)
    ax.set_xlabel("Lag time (fs)")
    ax.set_ylabel("MSD")
    ax.legend(title="Atom type")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = trajectory_path.with_name(trajectory_path.stem + "_msd.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    output_paths = [plot_trajectory(*config) for config in INPUTS]
    for output_path in output_paths:
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
