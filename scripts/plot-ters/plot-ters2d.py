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
parser.add_argument("--molecule", action='store_true', help="Whether molecule should be shown on top of the image.")
parser.add_argument("--grid", action='store_true', help="Whether grid should be shown on top of the image.")
parser.add_argument("--intensity", type=str, default='yes', help="Whether intensity should be shown on top of the image.")
parser.add_argument("--interpolate", action='store_true', help="Whether image should be interpolated between grid positions.")
args = parser.parse_args()

### Read calculation setup
working_dir = Path("./ters2d")
control_file = next(working_dir.glob("mode*/tippos*/positive_displacement/field_on/control.in"))
tip_height = None
with control_file.open() as f:
    for line in f:
        if line.strip().startswith("tip_molecule_distance"):
            tip_height = float(line.split()[-1])

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
                values = line.split('=')[-1].split('#')[0].strip().strip('(),')
                xmin, xmax, ymin, ymax = map(float, values.split(','))
else:
    print("run-ters.py not found in working directory. Please add this file with the scan_range = (xmin, xmax, ymin, ymax) tags.")
    exit(1)


fig, ax = plt.subplots()

### Plot molecule on top
# Read system
unconstrained_geometry_file = "geometry_unconstrained.in"
mol_system = ase.io.read(Path(unconstrained_geometry_file))
periodic = mol_system.pbc.all()
positions = mol_system.get_positions()
numbers = mol_system.get_atomic_numbers()

# Plot
plot_mol = args.molecule
if plot_mol:
    ax.scatter(positions[:, 0], positions[:, 1],
        c=[jmol_colors[n] for n in numbers], s=[covalent_radii[n] * 100 for n in numbers],
        edgecolors='k', linewidths=0.3, zorder=1)


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


### Read and save intensity data
mode_idx = args.mode
mode_dir = working_dir / f'mode_{mode_idx:03d}'


filepath = Path(f"rawdata/intensity_{mode_idx:03d}.dat")
outdir = Path("rawdata")
outdir.mkdir(parents=True, exist_ok=True)

try:
    ters = ffters.analyze_2d_ters(
        working_dir=Path('./ters2d'),
        mode_idx=[mode_idx],
        efield=efield,
        dq=dq,
        periodic=periodic,
        no_groundstate=True
    )

    np.savetxt(filepath, ters['intensity'])
    print(f"Saved intensity data to {filepath}")

    intensity_available = True

except FileNotFoundError:
    print(f"Intensity data not found for mode {mode_idx}")
    intensity_available = False


### Plot grid
coords, indices = read_grid_coords(mode_dir)
n_grid_points = len(coords)
print(f"Number of grid points: {n_grid_points}")

plot_grid = args.grid

if plot_grid:
    ax.scatter(coords[:, 0], coords[:, 1], s=50, color='black', zorder=3)

    for (x, y), pos_idx in zip(coords, indices):
        ax.annotate(
            str(pos_idx),
            (x, y),
            textcoords='offset points',
            xytext=(4, 4),
            fontsize=6,
            color='black'
        )



### Only continue if intensity exists
if intensity_available:

    ### Find nearby modes
    f = freqs[mode_idx]

    selected_modes = {mode_idx}

    print(f"Mode {mode_idx}: {f:.3f} 1/cm")

    width = args.wth

    close_modes = [
        j for j, fj in enumerate(freqs)
        if j != mode_idx and abs(fj - f) < width
    ]

    if close_modes:
        print(f"  → Modes within {width} 1/cm: {close_modes}")
        selected_modes.update(close_modes)

    new_mods = sorted(selected_modes)
    new_freqs = [freqs[i] for i in new_mods]

    ### Read and treat data
    files = [filepath for f in new_freqs]

    ters_intensity = read_and_sum_intensities(files)

    tippos_indices = ters['tippos_indices']

    intensity_per_tippos = {
        idx: val
        for idx, val in zip(tippos_indices, ters_intensity)
    }

    valid = np.isin(indices, list(intensity_per_tippos.keys()))

    xs = coords[valid, 0]
    ys = coords[valid, 1]

    vals = np.array([
        intensity_per_tippos[idx]
        for idx in indices[valid]
    ])

    xi = np.unique(xs)
    yi = np.unique(ys)

    ### Plot intensity data
    # Plot
    plot_intensity = args.intensity
    if plot_intensity=='yes':
        interpolate = args.interpolate
        if interpolate:
            from scipy.interpolate import griddata
            nbins = 100
            xi = np.linspace(xmin, xmax, nbins)
            yi = np.linspace(ymin, ymax, nbins)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((xs, ys), vals, (Xi, Yi), method='linear')
        else:
            ix = np.argmin(np.abs(xi[:, None] - xs[None, :]), axis=0)
            iy = np.argmin(np.abs(yi[:, None] - ys[None, :]), axis=0)
            Zi = np.full((len(yi), len(xi)), np.nan)
            Zi[iy, ix] = vals
            # for plotting 1D arrays
            if len(np.unique(ys)) == 1:
                yi = np.array([yi[0] - 1.2, yi[0] + 1.2])
                Zi = np.tile(Zi[0], (2, 1))
            elif len(np.unique(xs)) == 1:
                xi = np.array([xi[0] - 1.2, xi[0] + 1.2])
                Zi = np.tile(Zi[:, 0:1], (1, 2))

        cmap = plt.cm.viridis.copy()
        im = ax.pcolormesh(xi, yi, np.ma.masked_invalid(Zi), cmap=cmap, shading='auto', vmin=0, vmax=None, zorder=0)
        plt.colorbar(im, ax=ax)

ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.set_xlabel(r'$x$ [$\mathrm{\AA}$]')
ax.set_ylabel(r'$y$ [$\mathrm{\AA}$]')
ax.grid(False)
plt.tight_layout()

# Title and Saving
save_dir = Path("images")
save_dir.mkdir(parents=True, exist_ok=True)
if intensity_available:
    ax.set_title(rf'TERS image, f = {f:.3f} 1/cm ' rf'({min(new_mods):.0f}-{max(new_mods):.0f})')
    outfile = save_dir / f"2d_m{args.mode}_w{width:.0f}.png"
else:
    ax.set_title(f'Grid for mode {mode_idx}')
    outfile = save_dir / f"grid_m{args.mode}.png"

plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f"Saved figure to {outfile}")

#plt.close()
plt.show()
