from ase.io import read

mol = read("fepc.vasp")

# Rotate CO
mol.rotate(0, 'z', center='COM')

mol.write("fepcrot.vasp")
