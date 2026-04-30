from pathlib import Path
import argparse
import numpy as np
import ase.io
import sys
import os
sys.path.append(os.path.expanduser("~/.local/bin"))
import finite_field_ters as ffters
import ast
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
from ase.data.colors import jmol_colors
from ase.data import covalent_radii

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
parser.add_argument("--rot", type=float, default=0, help="Angle by which the image should be rotated.")
parser.add_argument("--plot_mol", type=bool, default=False, help="Whether molecule should be shown on top of the image.")
parser.add_argument("--grid", type=ast.literal_eval, default=[], help="Plot image as part of a bigger grid: [nbinsx, nbinsy, xscan, yscan]. xyscan are the postions where first pixel is placed.")
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
dq = 5e-3
efield = -1e-1
nbins = None
if runters_file.exists():
    with runters_file.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("dq"):
                dq = float(line.split('=')[-1].strip().rstrip(','))
            elif line.startswith("efield"):
                efield = float(line.split('=')[-1].strip().rstrip(','))
            elif line.startswith("bins"):
                value_str = line.split('=')[-1].strip()
                nbins = tuple(int(x) for x in ast.literal_eval(value_str))
            elif line.startswith("scan_range"):
                values = line.split('=')[-1].strip().strip('(),')
                xmin, xmax, ymin, ymax = map(float, values.split(','))

else:
    print("run-ters.py not found in working directory. Please add this file with the nbins = (n,m) and scan_range = (xmin, xmax, ymin, ymax) tags.")
    exit(1)

print(f"nbins = {nbins}")
print(f"xmin = {xmin}")
print(f"xmax = {xmax}")
print(f"ymin = {ymin}")
print(f"ymax = {ymax}")

ex = np.array([xmin, xmax, ymin, ymax])
unconstrained_geometry_file = "geometry_unconstrained.in"
mol_system = ase.io.read(Path(unconstrained_geometry_file))
periodic = mol_system.pbc.all()
positions = mol_system.get_positions()
numbers = mol_system.get_atomic_numbers()

mode_idx = [args.mode]
xyz_file = next(Path(".").glob(args.xyzfile))  # pick the first match
freqs = []

with xyz_file.open() as f:
    for line in f:
        line = line.strip()
        if "stable frequency at" in line:
            parts = line.split()
            freq = float(parts[3])
            freqs.append(freq)

### Save raw data
filepath = Path(f"rawdata/intensity_{mode_idx[0]}.dat")
if not filepath.exists():
    ters = ffters.analyze_2d_ters(working_dir=Path('./ters2d'), mode_idx = mode_idx, efield=efield, dq=dq, nbins=nbins, periodic=periodic, no_groundstate=True)
    outdir = Path("rawdata")
    outdir.mkdir(parents=True, exist_ok=True)
    np.savetxt(filepath, ters['intensity'])
    print("Saved intensity data to rawdata folder")

### Find nearby modes to be plotted
idx = mode_idx[0]
f = freqs[idx]
selected_modes = set()
selected_modes.add(idx)
print(f"Mode {idx}: {f:.3f} 1/cm")
width = args.wth   # set your window here
close_modes = [
    j for j, fj in enumerate(freqs)
    if j != idx and abs(fj - f) < width
]
if close_modes:
    print(f"  →  Modes within {width} 1/cm: {close_modes}")
    selected_modes.update(close_modes)
        
new_mods = sorted(selected_modes)
new_freqs = [freqs[i] for i in new_mods]


### Read and treat data
files = [f"rawdata/intensity_{idx}.dat" for f in new_freqs]
ters_intensity = read_and_sum_intensities(files)

# Rotate
angle = args.rot
ters_intensity = rotate(ters_intensity, angle=angle, reshape=False, order=1, mode='nearest')

# Crop
# Beware if image is cropped grid may not match actual pixel size used in calculations
Ny, Nx = ters_intensity.shape
x = np.linspace(ex[0], ex[1], Nx)
y = np.linspace(ex[2], ex[3], Ny)
ix = np.where((x >= xmin) & (x <= xmax))[0]
iy = np.where((y >= ymin) & (y <= ymax))[0]
ters_intensity = ters_intensity[iy[0]:iy[-1]+1, ix[0]:ix[-1]+1]
ex = [x[ix[0]], x[ix[-1]], y[iy[0]], y[iy[-1]]]


### Plot molecule on top
unconstrained_geometry_file = "geometry_unconstrained.in"
mol_system = ase.io.read(Path(unconstrained_geometry_file))
periodic = mol_system.pbc.all()
positions = mol_system.get_positions()
numbers = mol_system.get_atomic_numbers()

# Rotation
theta = np.deg2rad(angle)
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])
center = positions[:, :2].mean(axis=0)
pos_rot = (positions[:, :2] - center) @ R.T + center

# Plot
plot_mol = args.plot_mol
if plot_mol:
    fig, ax = plt.subplots()
    ax.scatter(
        pos_rot[:, 0],
        pos_rot[:, 1],
        c=[jmol_colors[n] for n in numbers],
        s=[covalent_radii[n] * 100 for n in numbers],
        edgecolors='k',
        linewidths=0.3,
        zorder=1
    )

### Plot
# Add a grid
nx, ny = nbins[0], nbins[1] 
if args.grid != []:
    nx, ny = int(args.grid[0]), int(args.grid[1])

xbins = np.linspace(xmin, xmax, nx + 1)
ybins = np.linspace(ymin, ymax, ny + 1)
xcenters = 0.5 * (xbins[:-1] + xbins[1:])
ycenters = 0.5 * (ybins[:-1] + ybins[1:])
grid = np.full((ny, nx), np.nan)

# Determine insertion point
xscan, yscan = np.min(xcenters), np.min(ycenters)
if args.grid != []:
    xscan, yscan = int(args.grid[2]), int(args.grid[3])

iy = np.argmin(np.abs(ycenters - yscan))
ix = np.argmin(np.abs(xcenters - xscan))

# Insert data
data = ters_intensity
if data.ndim == 1:
    n = len(data)
    ix_end = min(nx, ix + n)
    grid[iy, ix:ix_end] = data[:ix_end - ix]

elif data.ndim == 2:
    ny_p, nx_p = data.shape
    iy_end = min(ny, iy + ny_p)
    ix_end = min(nx, ix + nx_p)
    grid[iy:iy_end, ix:ix_end] = data[:iy_end - iy, :ix_end - ix]

masked = np.ma.masked_invalid(grid)

# Plot
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='grey')

im = ax.imshow(masked, origin='lower', extent=ex, cmap=cmap, interpolation='none')

ax.set_xlabel(r'$x$ [$\mathrm{\AA}$]')
ax.set_ylabel(r'$y$ [$\mathrm{\AA}$]')
ax.set_title(rf'TERS image, f = {f:.3f} 1/cm ({min(new_mods):.0f}-{max(new_mods):.0f})')

ax.set_xticks(xbins[::max(1, nx // 10)])
ax.set_yticks(ybins[::max(1, ny // 10)])
ax.set_xticks(xbins, minor=True)
ax.set_yticks(ybins, minor=True)

ax.grid(which='minor', color='white', linewidth=0.5)
plt.colorbar(im, ax=ax)
plt.tight_layout()

# Save
save_dir = Path("images")
save_dir.mkdir(parents=True, exist_ok=True)
outfile = save_dir / f"2d_m{args.mode}_w{width:.0f}.png"
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.show()
