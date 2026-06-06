import pickle
from pathlib import Path
import numpy as np, glob
from ase.io import read, write

import sys
import os
import shutil

sys.path.append(os.path.expanduser("~/.local/bin"))
from finite_field_ters import FiniteFieldTERS

species_dir = "/home/planelp1/species_defaults/light/" # Triton
# species_dir = "/projappl/project_2001912/species_defaults/light/" # CSC


def find_species_file(symbol, species_dir):
    for root, _, files in os.walk(species_dir):
        for f in files:
            parts = f.split("_")
            if len(parts) >= 2 and parts[1] == symbol:
                return os.path.join(root, f)

    return None

### Consider only unconstrained atoms
with open("geometry.in", "r") as f:
    lines = f.readlines()

lines_unconstrained = []
lines_constrained = []
symbols_unconstrained = []
symbols_constrained = []
for line in lines:
    if "constrain_relaxation" in line:
        prev_line = lines_unconstrained.pop() 
        lines_constrained.append(prev_line)  # Save popped line
        lines_constrained.append(line)
        continue
    lines_unconstrained.append(line)

# Save all popped lines to geo.in
geo_constrained = "geometry_constrained.in"
with open(geo_constrained, "w") as f:
    f.writelines(lines_constrained)


geo_unconstrained = "geometry_unconstrained.in"
with open(geo_unconstrained, "w") as f:
    f.writelines(lines_unconstrained)

### Read masses of constrained atoms
for line in lines_constrained:
    if line.strip().startswith("atom"):
        symbol = line.split()[4] 
        symbols_constrained.append(symbol)

### Read masses of unconstrained atoms
for line in lines_unconstrained:
    if line.strip().startswith("atom"):
        symbol = line.split()[4]
        symbols_unconstrained.append(symbol)

print("Unconstrained symbols:", set(symbols_unconstrained))
print("Constrained symbols:", set(symbols_constrained))

mass_dict = {}
masses = []
# masses are read from species_defaults                                     !!!
for symbol in symbols_unconstrained:
    file_path = find_species_file(symbol, species_dir)
    with open(file_path, "r") as elem_file:
        for line in elem_file:
            if "mass" in line:
                mass = float(line.split()[-1])
                mass_dict[symbol] = mass
                break

for symbol in symbols_unconstrained:
    masses.append(mass_dict[symbol])
masses = np.array(masses)
print("Unconstrained atoms masses: ", mass_dict)

### Check molecule is centered in (0,0), assuming molecule is only the unconstrained atoms
# This will fail if one of the unconstrained molecule atoms is also constrained in the slab!!!
unconstrained_atoms = read(geo_unconstrained)
com_unconstrained = unconstrained_atoms.get_center_of_mass()
if not np.allclose(com_unconstrained, [0, 0, 0], atol=1e-2):
    print(f"WARNING: molecule COM may not at origin! Offset = {com_unconstrained}")

    atoms = read('geometry.in')
    atoms.translate([-com_unconstrained[0], -com_unconstrained[1], -com_unconstrained[2]])
    write('geometry_centered_all.in', atoms)
    print("Centered structure saved in geometry_centered_all.in, will not be used, rename geometry.in and rerun if you wish to use it")

### Read Hessian
files = glob.glob("massweighted_Hessian.*.dat")
if not files:
    raise FileNotFoundError("No file matching massweighted_Hessian.*.dat found")
h = np.loadtxt(files[0])

__all__ = [
    'h', 'masses', 'geo_unconstrained', 'geo_constrained',
    'species_dir', 'symbols_constrained', 'symbols_unconstrained',
    'find_species_file'
]
