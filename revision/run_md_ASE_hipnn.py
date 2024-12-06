import argparse
import os
import torch

from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import Trajectory
from ase import units

import hippynn
from hippynn.experiment.serialization import load_model_from_cwd
from hippynn.interfaces.ase_interface import HippynnCalculator

def main():
    """Run Langevin MD using prtrained HIPPYNN model.

    Args:
        model_location: Path to the directory containing the model file.
        molecule: Name of the molecule file (e.g. C7eq-TT.xyz).
        steps: Number of steps for the MD simulation (default: 10000).

    Load ML potential, assign as ASE calculator to ASE atoms (your molecule)
    Write simple log file 
    """
    parser = argparse.ArgumentParser(description='Run Langevin dynamics on a molecule')
    parser.add_argument('--model_location', type=str, help='Path to the model file')
    parser.add_argument('--molecule', type=str, help='Name of the molecule file (e.g. C7eq-TT.xyz)')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps for MD')
    args = parser.parse_args()

    # Load the model
    current_dir = os.getcwd()
    os.chdir(args.model_location)
    model = load_model_from_cwd(map_location=torch.device('cpu')) # CPU runs for small molecules usually faster
    predictor = hippynn.graphs.Predictor.from_graph(model) 
    predictor.to(torch.float64)
    energy_node = model.node_from_name("T") # T = energy (varies by setup defined during training)
    calc = HippynnCalculator(energy_node)   # create ASE calculator object
    calc.to(torch.float64)
    os.chdir(current_dir) 

    # Load the molecule
    molecule_file = os.path.join(args.molecule)
    atoms = read(molecule_file)

    # assign hippynn as calculator
    atoms.calc = calc

    # Set initial velocities corresponding to 300 K
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    # Define the Langevin dynamics parameters
    # see ASE docs and typical MD for alanine dipeptide - a lot of sources 
    timestep = 2 * units.fs  # Time step of 2 femtoseconds
    temperature = 300  # Temperature in Kelvin (close to body temperature)
    friction = 0.1 / units.fs  # Friction coefficient in atomic units

    # Initialize Langevin dynamics
    dyn = Langevin(atoms, timestep, temperature_K=temperature, friction=friction)

    # Create trajectory file name based on the molecule name
    traj_filename = f"langevin_300K_{args.molecule}.traj"
    traj = Trajectory(traj_filename, "w", atoms)
    dyn.attach(traj.write, interval=100)

    # Function to print energies during the simulation
    log_filename = f"energy_log_{args.molecule}.txt"
    with open(log_filename, "w") as log_file:
        def log_energy(a=atoms):
            epot = a.get_potential_energy() / len(a)  # Potential energy per atom
            ekin = a.get_kinetic_energy() / len(a)  # Kinetic energy per atom
            temp = ekin / (1.5 * units.kB)  # Temperature from kinetic energy
            log_file.write(f"Energy per atom: Epot={epot:.3f} eV Ekin={ekin:.3f} eV (T={temp:.1f} K)\n")

        dyn.attach(log_energy, interval=100)

    # Run the dynamics for the specified number of steps
    print("Starting Langevin dynamics...")
    log_energy()
    dyn.run(args.steps)


if __name__ == '__main__':
    main()