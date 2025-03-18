# %%
from ase.io import read, write
import os
from pathlib import Path
from ase.io.gaussian import write_gaussian_in
import argparse
# %%
# %%
def write_Gaussian_inputs(xyz_file, args):
    atoms = read(xyz_file)
    atoms.pbc = False
    with open(f'{xyz_file.stem}.com', 'w+') as com:
        write_gaussian_in(fd = com,
              method = args.method,
              extra = args.extra,
              basis = args.basis,
              properties = args.properties,
              charge = args.charge,
              mult = args.mult,
              atoms = atoms,
              **{'mem' : args.mem, 'cpu' : args.cpu})
# %%
def main():
    """main function for writing Gaussian inputs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path, help='path to look for xyz files')
    parser.add_argument('--method', type=str, default='wb97x', help='method to use')
    parser.add_argument('--extra', type=str, default='scf=tight', help='extra options')
    parser.add_argument('--basis', type=str, default='6-31G*', help='basis set')
    parser.add_argument('--properties', type=str, nargs='+', default=['energy', 'forces'], help='properties to calculate')
    parser.add_argument('--charge', type=int, default=0, help='charge')
    parser.add_argument('--mult', type=int, default=1, help='multiplicity')
    parser.add_argument('--mem', type=str, default='32GB', help='memory')
    parser.add_argument('--cpu', type=str, default='0-15', help='cpu')
    args = parser.parse_args()

    for f in args.path.rglob('*.xyz'): # recursively for subfolders
        write_Gaussian_inputs(xyz_file = f, args=args)

if __name__ == "__main__":
    main()


# %%
