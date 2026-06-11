from ase.io.cube import read_cube_data
from ase.io import write

data1, atoms1 = read_cube_data('01nocharge_diff.cube')
data2, atoms2 = read_cube_data('10_diff.cube')
diff = data2 - data1

# Save the difference as a new cube# Attach the volumetric data to the Atoms object
write('1001nocharge_diff.cube', atoms2, data=diff)

print("Saved cube difference to a '.cube'")

# Now you have the data array and ASE Atoms object
print("Grid shape 1:", data1.shape)
print("Grid shape 2:", data2.shape)
print("Atoms 1:", atoms1)
print("Atoms 2:", atoms2)
