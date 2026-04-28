from ase.io import read, write
import numpy as np

atoms = read("POSCAR")

# move based on com
com = atoms.get_center_of_mass()
target_com = np.array([0,0,0])
shift = target_com - com

# or choose shift
shift = np.array([0, 0, 0])

# Move all atoms
atoms.positions += shift

# Save the moved molecule
write("POSCAR_moved", atoms)

