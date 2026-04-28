from ase.io import read
from ase.optimize import BFGS, FIRE
from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.md import MDLogger

from ase.units import kB

import sys, os
import numpy as np
from pathlib import Path
import argparse

import torch

def save_traj_in_xyz(traj):
    frames = read(traj,index=':')
    write(traj.with_suffix('.xyz'), frames)
    return 

class MiniMD:

    def __init__(self, model='mace', size=None, config=None, temperature=300, timefs=1.0, fmax=2.0, steps=1000):

        self.wkdir = Path(model)
        self.fmax = fmax                   # eV/A
        self.time_step = timefs * units.fs # 1 femtosecond
        self.model = model
        self.steps = steps
        self.temperature = temperature      # K

        try:
            self.wkdir.mkdir(parents=True, exist_ok=True)
            print(f"Directory '{self.wkdir}' created successfully.")
        except FileExistsError:
            print(f"Directory '{self.wkdir}' already exists.")

        self.log = open(self.wkdir/'minimd.log','w')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Running on: {self.device}")

        self.md_traj = self.wkdir/'md.traj'
        self.relax_traj = self.wkdir/'relax.traj'

        print(f"reading atoms from : {config}")

        if 'lammpstrj' in config:
            self.atoms = read(config, format='lammps-data')
        else:
            self.atoms = read(config)

        if self.model == 'mace':
            from mace.calculators import mace_mp
            self.atoms.calc = mace_mp(model='medium',device=self.device, default_dtype='float64')

        elif self.model == 'nequip':
            from nequip.ase import NequIPCalculator
            self.atoms.calc = NequIPCalculator.from_compiled_model(
                compile_path="NequIP-OAM-L-0.1.nequip.pth",
                device=self.device, 
            )

        elif self.model == 'allegro':
            from nequip.ase import NequIPCalculator
            self.atoms.calc = NequIPCalculator.from_compiled_model(
                compile_path="Allegro-OAM-L-0.1.nequip.pth", 
                device=self.device, 
            )

        elif self.model == 'mattersim':
            from mattersim.forcefield import MatterSimCalculator
            self.atoms.calc = MatterSimCalculator(device=self.device)

        elif self.model == 'sevennet':
            from sevenn.calculator import SevenNetCalculator
            self.atoms.calc = SevenNetCalculator(model='7net-omni', modal='mpa')

        elif self.model == 'upet':
            from upet.calculator import UPETCalculator
            self.atoms.calc = UPETCalculator(
                    model="pet-omat-m",
                    version="1.0.0", 
                    device=self.device, 
                    non_conservative=False)

        elif self.model == 'uma':

            os.environ['CXX'] = 'g++'

            from fairchem.core import FAIRChemCalculator
            from fairchem.core.units.mlip_unit import load_predict_unit
            predictor = load_predict_unit(
                    path="./uma-s-1p1.pt", 
                    device=self.device,
                    inference_settings="turbo"
                    )

            calc = FAIRChemCalculator(predictor, task_name="omat")
            self.atoms.calc = calc
        else:
            raise NotImplementedError("more models to come, but not yet.")

        return

# Initial Geometry Optimization
    def relax(self):

        print("Starting Structure Optimization...")
        relax = FIRE(self.atoms)
        traj = Trajectory(self.relax_traj, 'w', self.atoms)
        relax.attach(traj.write, interval=1)
        relax.run(fmax=self.fmax)
        traj.close()

        save_traj_in_xyz(self.relax_traj)

# Setup Molecular Dynamics (Langevin Thermostat)
    def thermalize(self):

        print("Starting Molecular Dynamics...")
        dyn = Langevin(self.atoms, 
            timestep=self.time_step,
            temperature_K=self.temperature,
            friction=0.01,
            logfile=self.wkdir/'md.log',
            )

        traj = Trajectory(self.md_traj, 'w', self.atoms)
        dyn.attach(traj.write, interval=10)
        dyn.run(self.steps)
        traj.close()

        save_traj_in_xyz(self.md_traj)

        return 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=str, default='sio2.xyz')
    parser.add_argument('-m','--model', type=str, default='mace')
    parser.add_argument('-f','--fmax', type=float, default=2.0)
    parser.add_argument('-s','--steps', type=int, default=1000)
    parser.add_argument('-t','--temperature', type=float, default=300.0)
    args = parser.parse_args()

    md = MiniMD(config = args.config, 
                model = args.model, 
                fmax = args.fmax, 
                temperature = args.temperature, 
                steps = args.steps)

    md.relax()
    md.thermalize()

