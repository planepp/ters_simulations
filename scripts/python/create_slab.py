# Create slab
from ase.build import fcc100, fcc110, fcc111
ag100 = fcc100('Ag', a=4.2, size=(8,8,4), vacuum=100)

from ase.io import write
ag100.write('Ag100.aims', format='aims')
