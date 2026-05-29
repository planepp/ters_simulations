#!/usr/bin/env python3
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
parser.add_argument("--scan_path", type=str, default='./ters2d', help="Path to the simulation result (usually 'ters2d')")
parser.add_argument("--modes", type=int, nargs='+', default=None, help="Additional mode indices to sum over (e.g. --modes 5 6 7). The primary mode is always included.")
parser.add_argument("--molecule", action='store_true', help="Whether molecule should be shown on top of the image.")
parser.add_argument("--grid", action='store_true', help="Whether grid should be shown on top of the image.")
parser.add_argument("--intensity", type=str, default='yes', help="Whether intensity should be shown on top of the image.")
parser.add_argument("--interpolate", action='store_true', help="Whether image should be interpolated between grid positions.")
parser.add_argument("--dq", type=float, default=5e-3, help="dq used in the simulation")
parser.add_argument("--efield", type=float, default=-1e-1, help="Electric field used in the simulation")
args = parser.parse_args()

dq = args.dq
efield = args.efield

### Read calculation setup
working_dir = Path(args.scan_path)
all_control_files = working_dir.glob(f"mode_{args.mode:03d}/tippos*/positive_displacement/field_on/control.in")
tip_height = None
xmin, xmax, ymin, ymax = (0,0,0,0)
with next(all_control_files).open() as f:
    for line in f:
        if line.strip().startswith("tip_molecule_distance"):
            tip_height = float(line.split()[-1])

# gets tippos boundaries
for control_file in all_control_files:
    with control_file.open() as f:
        for line in f:
            if line.startswith("rel_shift_from_tip"):
                x, y = [float(x) for x in line.split()[1:]]
                if x < xmin:
                    xmin = x
                if x > xmax:
                    xmax = x
                if y < ymin:
                    ymin = y
                if y > ymax:
                    ymax = y
                break
print("Found x and y min and max in all tippos:", xmin, xmax, ymin, ymax)


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
        working_dir=Path(args.scan_path),
        mode_idx=[mode_idx],
        efield=efield,
        dq=dq,
        periodic=periodic,
        no_groundstate=True
    )

    #print(ters['intensity'])
    np.savetxt(filepath, ters['intensity'])
    print(f"Saved intensity data to {filepath}")

    intensity_available = True

except FileNotFoundError as e:
    print(f"Intensity data not found for mode {mode_idx}: {e}")
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

    ### Collect selected modes
    selected_modes = {mode_idx}
    if args.modes:
        selected_modes.update(args.modes)

    new_mods = sorted(selected_modes)
    new_freqs = [freqs[i] for i in new_mods]

    print(f"Summing over modes: {new_mods}")
    for m in new_mods:
        print(f"  Mode {m}: {freqs[m]:.3f} 1/cm")

    ### Load intensity data per mode
    # Store each mode's (xs, ys, vals) separately so that interpolation can be
    # done per-mode before summing. Summing the combined sparse grid first and
    # then interpolating is wrong: a dark point from one mode that has no
    # counterpart in another mode would incorrectly suppress the interpolated
    # surface in that region.
    per_mode_data = []   # list of (xs, ys, vals) one entry per mode

    for m in new_mods:
        m_filepath = Path(f"rawdata/intensity_{m:03d}.dat")
        m_mode_dir = working_dir / f'mode_{m:03d}'
        m_coords, m_indices = read_grid_coords(m_mode_dir)

        # Load from disk if already saved, otherwise compute and save
        if m_filepath.exists():
            m_intensity = np.loadtxt(m_filepath)
            m_tippos = m_indices   # sorted order matches analyze_2d_ters
        else:
            try:
                ters_m = ffters.analyze_2d_ters(
                    working_dir=Path('./ters2d'),
                    mode_idx=[m],
                    efield=efield,
                    dq=dq,
                    periodic=periodic,
                    no_groundstate=True
                )
                np.savetxt(m_filepath, ters_m['intensity'])
                print(f"Saved intensity data to {m_filepath}")
                m_intensity = ters_m['intensity']
                m_tippos = ters_m['tippos_indices']
            except FileNotFoundError:
                print(f"Intensity data not found for mode {m}, skipping.")
                continue

        # Map tippos index -> (coord, intensity)
        tippos_indices      = ters['tippos_indices']
        tippos_to_coord     = dict(zip(m_indices, m_coords))
        tippos_to_intensity = dict(zip(tippos_indices, m_intensity))

        valid  = [idx for idx in tippos_indices if idx in tippos_to_coord and idx in tippos_to_intensity]
        m_xs   = np.array([tippos_to_coord[idx][0]  for idx in valid])
        m_ys   = np.array([tippos_to_coord[idx][1]  for idx in valid])
        m_vals = np.array([tippos_to_intensity[idx] for idx in valid])

        per_mode_data.append((m_xs, m_ys, m_vals))

    ### Plot intensity data
    plot_intensity = args.intensity
    if plot_intensity == 'yes':
        interpolate = args.interpolate
        if interpolate:
            # Interpolate each mode onto the fine grid independently, then sum.
            # This is correct when modes have different or offset grids: each
            # mode contributes its own smooth surface, and dark/missing points
            # in one mode don't suppress the combined image.
            from scipy.interpolate import griddata
            nbins = 100
            xi = np.linspace(xmin, xmax, nbins)
            yi = np.linspace(ymin, ymax, nbins)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi_total   = np.zeros((nbins, nbins))
            any_coverage = np.zeros((nbins, nbins), dtype=bool)
            for m_xs, m_ys, m_vals in per_mode_data:
                Zi_m = griddata((m_xs, m_ys), m_vals, (Xi, Yi), method='linear')
                covered = ~np.isnan(Zi_m)
                any_coverage |= covered
                Zi_total += np.nan_to_num(Zi_m, nan=0.0)
            Zi = np.where(any_coverage, Zi_total, np.nan)
        else:
            # Non-interpolated: sum intensities at each measured tip position.
            # Combine all modes into a single sparse grid for display.
            combined_intensity = {}
            all_coords_map = {}
            for m_xs, m_ys, m_vals in per_mode_data:
                for x, y, v in zip(m_xs, m_ys, m_vals):
                    # use rounded coord as key to merge coincident points
                    key = (round(float(x), 6), round(float(y), 6))
                    all_coords_map[key] = (x, y)
                    combined_intensity[key] = combined_intensity.get(key, 0.0) + v

            xs   = np.array([all_coords_map[k][0] for k in combined_intensity])
            ys   = np.array([all_coords_map[k][1] for k in combined_intensity])
            vals = np.array([combined_intensity[k] for k in combined_intensity])

            xi = np.unique(xs)
            yi = np.unique(ys)
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
        im = ax.pcolormesh(xi, yi, np.ma.masked_invalid(Zi), cmap=cmap, shading='auto', vmin=None, vmax=None, zorder=0)
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
    freq_str = ", ".join(f"{freqs[m]:.1f}" for m in new_mods)
    modes_str = "-".join(str(m) for m in new_mods)
    ax.set_title(rf'TERS image, modes {modes_str} ({freq_str} 1/cm)')
    outfile = save_dir / f"2d_m{modes_str}.png"
else:
    ax.set_title(f'Grid for mode {mode_idx}')
    outfile = save_dir / f"grid_m{args.mode}.png"

plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f"Saved figure to {outfile}")

#plt.close()
plt.show()
