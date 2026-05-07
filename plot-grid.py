from pathlib import Path
import argparse
import numpy as np
import ase.io
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
from ase.data.colors import jmol_colors
from ase.data import covalent_radii

def read_grid_coords(mode_dir: Path):
    """
    Read tip positions from all tippos_*/positi/fieldon/controlin/ folders.
    Returns array of shape (n_points, 2) with (x, y) coordinates,
    and the corresponding tippos index for each point.
    """
    coords = []
    indices = []

    for tippos_dir in sorted(mode_dir.glob('tippos_*')):
        control_file = tippos_dir / 'positive_displacement' / 'field_on' / 'control.in'
        with open(control_file) as f:
            for line in f:
                if line.strip().startswith('rel_shift_from_tip'):
                    parts = line.split()
                    x, y = float(parts[1]), float(parts[2])
                    coords.append((x, y))
                    indices.append(int(tippos_dir.name.split('_')[1]))
                    break

    return np.array(coords), np.array(indices)

runters_file = Path("run-ters.py")
if runters_file.exists():
    with runters_file.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("scan_range"):
                values = line.split('=')[-1].strip().strip('(),')
                xmin, xmax, ymin, ymax = map(float, values.split(','))
else:
    print("run-ters.py not found in working directory. Please add this file with the nbins = (n,m) and scan_range = (xmin, xmax, ymin, ymax) tags.")
    exit(1)

parser = argparse.ArgumentParser(description="Calculate and plot a 2D TERS image")
parser.add_argument("mode", type=int, default=None, help="Mode index to process: single integer")
parser.add_argument("--plot_mol", type=bool, default=False, help="Whether molecule should be shown on top of the image.")
args = parser.parse_args()

mode = args.mode
mode_dir = Path(f'ters2d/mode_{mode:03d}')
coords, indices = read_grid_coords(mode_dir)

fig, ax = plt.subplots()
ax.scatter(coords[:, 0], coords[:, 1], s=50, color='black', zorder=3)
for (x, y), idx in zip(coords, indices):
    ax.annotate(str(idx), (x, y), textcoords='offset points', xytext=(4, 4), fontsize=6, color='black')

ax.set_xlabel(r'$x$ [$\mathrm{\AA}$]')
ax.set_ylabel(r'$y$ [$\mathrm{\AA}$]')
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.set_title(f'Tip position grid (mode {mode}) — {len(coords)} points')
ax.set_aspect('equal')
ax.grid(False)

### Plot molecule on top
unconstrained_geometry_file = "geometry_unconstrained.in"
mol_system = ase.io.read(Path(unconstrained_geometry_file))
periodic = mol_system.pbc.all()
positions = mol_system.get_positions()
numbers = mol_system.get_atomic_numbers()

plot_mol = args.plot_mol
if plot_mol:
    ax.scatter(positions[:, 0], positions[:, 1], c=[jmol_colors[n] for n in numbers], s=[covalent_radii[n] * 100 for n in numbers], edgecolors='k', linewidths=0.3, zorder=1)

plt.show()
