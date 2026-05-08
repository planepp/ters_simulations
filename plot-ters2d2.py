from pathlib import Path
import argparse
import numpy as np
import ase.io
import sys
import os
sys.path.append(os.path.expanduser("~/.local/bin"))
import finite_field_ters as ffters
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
from ase.data.colors import jmol_colors
from ase.data import covalent_radii

import ast
from scipy.ndimage import rotate

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

def compute_edges(centers):
    """Compute cell edges from bin centers (works for non-uniform spacing)."""
    centers = np.sort(centers)
    inner = 0.5 * (centers[:-1] + centers[1:])
    # extrapolate outer edges
    left  = centers[0]  - (centers[1]  - centers[0])  / 2
    right = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.concatenate([[left], inner, [right]])

def read_and_sum_intensities(files):
    if len(files) == 0:
        raise ValueError("File list is empty.")
    total_intensity = None
    for f in files:
        data = np.loadtxt(f)
        if total_intensity is None:
            total_intensity = data
        else:
            total_intensity += data
    return total_intensity


parser = argparse.ArgumentParser(description="Calculate and plot a 2D TERS image")
parser.add_argument("xyzfile", type=str, help="Path to the xyz file with all the normal frequencies")
parser.add_argument("mode", type=int, default=None, help="Mode index to process: single integer")
parser.add_argument("--wth", type=float, default=0, help="Width in 1/cm over which the nearby modes should be found and summed over.")
parser.add_argument("--plot_mol", type=bool, default=False, help="Whether molecule should be shown on top of the image.")
args = parser.parse_args()

### Read calculation setup
working_dir = Path("./ters2d")
control_file = next(working_dir.glob("mode*/tippos*/positive_displacement/field_on/control.in"))
tip_height = None
with control_file.open() as f:
    for line in f:
        if line.strip().startswith("tip_molecule_distance"):
            tip_height = float(line.split()[-1])
print(f"Tip-sample distance = {tip_height} Å")

runters_file = Path("run-ters.py")
if runters_file.exists():
    with runters_file.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("dq"):
                dq = float(line.split('=')[-1].strip().rstrip(','))
            elif line.startswith("efield"):
                efield = float(line.split('=')[-1].strip().rstrip(','))
            elif line.startswith("scan_range"):
                values = line.split('=')[-1].strip().strip('(),')
                xmin, xmax, ymin, ymax = map(float, values.split(','))
else:
    print("run-ters.py not found in working directory. Please add this file with the scan_range = (xmin, xmax, ymin, ymax) tags.")
    exit(1)


### Plot grid
mode_idx = args.mode
mode_dir = Path(f'ters2d/mode_{mode_idx:03d}')
coords, indices = read_grid_coords(mode_dir)

fig, ax = plt.subplots()
ax.scatter(coords[:, 0], coords[:, 1], s=50, color='black', zorder=3)
for (x, y), pos_idx in zip(coords, indices):
    ax.annotate(str(pos_idx), (x, y), textcoords='offset points', xytext=(4, 4), fontsize=6, color='black')

ax.set_xlabel(r'$x$ [$\mathrm{\AA}$]')
ax.set_ylabel(r'$y$ [$\mathrm{\AA}$]')
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.set_title(f'Tip position grid (mode {mode_idx}) — {len(coords)} points')
ax.set_aspect('equal')
ax.grid(False)

### Parse xyz file: find frequencies
xyz_file = next(Path(".").glob(args.xyzfile))
freqs = []
with xyz_file.open() as f:
    for line in f:
        line = line.strip()
        if "stable frequency at" in line:
            parts = line.split()
            freq = float(parts[3])
            freqs.append(freq)

### Save raw  intensity data
nbins = (1,1)
unconstrained_geometry_file = "geometry_unconstrained.in"
mol_system = ase.io.read(Path(unconstrained_geometry_file))
periodic = mol_system.pbc.all()
positions = mol_system.get_positions()
numbers = mol_system.get_atomic_numbers()

filepath = Path(f"rawdata/intensity_{mode_idx}.dat")
ters = ffters.analyze_2d_ters(working_dir=Path('./ters2d'), mode_idx = [mode_idx], efield=efield, dq=dq, nbins=nbins, periodic=periodic, no_groundstate=True)
outdir = Path("rawdata")
outdir.mkdir(parents=True, exist_ok=True)
np.savetxt(filepath, ters['intensity'])
print("Saved intensity data to rawdata folder")

### Find nearby modes to be plotted
f = freqs[mode_idx]
selected_modes = set()
selected_modes.add(mode_idx)
print(f"Mode {mode_idx}: {f:.3f} 1/cm")
width = args.wth   # set your window here
close_modes = [
    j for j, fj in enumerate(freqs)
    if j != mode_idx and abs(fj - f) < width
]
if close_modes:
    print(f"  →  Modes within {width} 1/cm: {close_modes}")
    selected_modes.update(close_modes)
        
new_mods = sorted(selected_modes)
new_freqs = [freqs[i] for i in new_mods]

### Read and treat dat
files = [f"rawdata/intensity_{mode_idx}.dat" for f in new_freqs]
ters_intensity = read_and_sum_intensities(files)

### Plot molecule on top
# Plot
plot_mol = args.plot_mol
if plot_mol:
    ax.scatter(positions[:, 0], positions[:, 1],
        c=[jmol_colors[n] for n in numbers], s=[covalent_radii[n] * 100 for n in numbers],
        edgecolors='k', linewidths=0.3, zorder=1)

### Plot
# Build grid from tip positions
mode_dir = working_dir / f'mode_{args.mode:03d}'
coords, indices = read_grid_coords(mode_dir)

### Plot
mode_dir = working_dir / f'mode_{args.mode:03d}'
coords, indices = read_grid_coords(mode_dir)

# Build (x, y, intensity) from scan points
tippos_indices = ters['tippos_indices']
intensity_per_tippos = {idx: val for idx, val in zip(tippos_indices, ters_intensity)}

xs, ys, vals = [], [], []
for tippos_idx, intensity in intensity_per_tippos.items():
    i = np.where(indices == tippos_idx)[0]
    if len(i) == 0:
        continue
    x, y = coords[i[0]]
    xs.append(x)
    ys.append(y)
    vals.append(intensity)

xs, ys, vals = np.array(xs), np.array(ys), np.array(vals)

cmap = plt.cm.viridis.copy()
#im = ax.tripcolor(xs, ys, vals, cmap=cmap)
from scipy.interpolate import LinearNDInterpolator

# Mirror points across all 4 edges
xs_m = np.concatenate([xs, 2*xmin - xs, 2*xmax - xs, xs,          xs         ])
ys_m = np.concatenate([ys, ys,           ys,           2*ymin - ys, 2*ymax - ys])
vals_m = np.tile(vals, 5)  # same values for all mirrors

# Interpolate on mirrored points
interp = LinearNDInterpolator(np.column_stack([xs_m, ys_m]), vals_m)

nx, ny = 200, 200
xi = np.linspace(xmin, xmax, nx)
yi = np.linspace(ymin, ymax, ny)
Xi, Yi = np.meshgrid(xi, yi)
Zi = interp(Xi, Yi)

im = ax.pcolormesh(xi, yi, Zi, cmap=cmap, shading='auto', vmin=0, vmax=None)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.set_xlabel(r'$x$ [$\mathrm{\AA}$]')
ax.set_ylabel(r'$y$ [$\mathrm{\AA}$]')
ax.set_title(rf'TERS image, f = {f:.3f} 1/cm ({min(new_mods):.0f}-{max(new_mods):.0f})')
ax.grid(False)
plt.colorbar(im, ax=ax)
plt.tight_layout()
# Save
save_dir = Path("images")
save_dir.mkdir(parents=True, exist_ok=True)
outfile = save_dir / f"2d_m{args.mode}_w{width:.0f}.png"
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.show()
