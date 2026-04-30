import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
import argparse
from scipy.ndimage import rotate
from pathlib import Path
import ase.io
import ast
from ase.data.colors import jmol_colors
from ase.data import covalent_radii

def parse_modes(s):
    """Parse mode argument: single number, comma-separated list, or range x-y."""
    modes = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            start, end = map(int, part.split("-"))
            modes.extend(range(start, end + 1))
        else:
            modes.append(int(part))
    return modes

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
parser.add_argument(
    "mode",
    type=str,
    default=None,
    help="Mode index(es) to process: single int, comma-separated list, or range (e.g., '7-31' or '7,17,23')"
)
parser.add_argument(
    "--wth",
    type=float,
    default=0,
    help="Width in 1/cm over which the nearby modes should be found and summed over."
)
parser.add_argument(
    "--rot",
    type=float,
    default=0,
    help="Angle by which the image should be rotated."
)
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

runters_file = Path("run-ters_triton.py")
dq = 5e-3
efield = -1e-1
nbins = None
if runters_file.exists():
    with runters_file.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("scan_range"):
                values = line.split('=')[-1].strip().rstrip(',')
                x1, x2, y1, y2 = map(float, values.strip('()').split(','))
            elif line.startswith("dq"):
                dq = float(line.split('=')[-1].strip().rstrip(','))
            elif line.startswith("efield"):
                efield = float(line.split('=')[-1].strip().rstrip(','))
            elif line.startswith("bins"):
                value_str = line.split('=')[-1].strip()
                nbins = tuple(int(x) for x in ast.literal_eval(value_str))

else:
    print("run-ters.py not found in working directory. Assuming default dq, efield.")
    print("run-ters.py not found in working directory. Please add this file with the nbins = (n,m) and xx, yy tags.")
    exit(1)

ex = np.array([x1, x2, y1, y2])


### Find what to plot
mode_idx = parse_modes(args.mode)
xyz_file = next(Path(".").glob(args.xyzfile))  # pick the first match
freqs = []

with xyz_file.open() as f:
    for line in f:
        line = line.strip()
        if "stable frequency at" in line:
            # extract the frequency (second-to-last token)
            parts = line.split()
            freq = float(parts[3])  # parts[3] is the number before "1/cm"
            freqs.append(freq)

# Find nearby modes
selected_freqs = [freqs[i] for i in mode_idx]
selected_modes = set()   # use a set to avoid duplicates
for idx in mode_idx:
    if 0 <= idx < len(freqs):
        f = freqs[idx]
        print(f"Mode {idx}: {f:.3f} 1/cm")
        selected_modes.add(idx)
        width = args.wth   # set your window here
        close_modes = [
            j for j, fj in enumerate(freqs)
            if j != idx and abs(fj - f) < width
        ]
        if close_modes:
            print(f"  →  Modes within {width} 1/cm: {close_modes}")
            selected_modes.update(close_modes)
    else:
        print(f"Invalid mode_idx {idx}, file has {len(freqs)} modes")
        
new_mods = sorted(selected_modes)
new_freqs = [freqs[i] for i in new_mods]


### Read and treat data
directory = "rawdata"
files = [
    f"rawdata/intensity_{f:.3f}.dat"
    for f in new_freqs
]
ters_intensity = read_and_sum_intensities(files).reshape(nbins[1], nbins[0])                            #################

# Rotate
angle = args.rot
ters_intensity = rotate(
    ters_intensity,
    angle=angle,        # degrees CCW
    reshape=False,   # keep same array size
    order=1,         # bilinear interpolation
    mode='nearest'   # avoid artificial zeros at edges
)

# Crop
# Beware if image is cropped grid may not match actual pixel size used in calculations
xmin, xmax = x1, x2
ymin, ymax = y1, y2
Ny, Nx = ters_intensity.shape
x = np.linspace(ex[0], ex[1], Nx)
y = np.linspace(ex[2], ex[3], Ny)
ix = np.where((x >= xmin) & (x <= xmax))[0]
iy = np.where((y >= ymin) & (y <= ymax))[0]
ters_intensity = ters_intensity[iy[0]:iy[-1]+1, ix[0]:ix[-1]+1]
ex = [x[ix[0]], x[ix[-1]], y[iy[0]], y[iy[-1]]]
print(ex)


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
'''
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
'''


### Plot
# Create full grid filled with NaN (empty)
full_grid = np.full((20, 20), np.nan)  # shape must be a tuple; fill with NaN not 0
xbins_centers = np.linspace(-6.5, 6.5, 20)
ybins_centers = np.linspace(-6.5, 6.5, 20)

y_scan, x_scan = 0, 0.1
iy = np.argmin(np.abs(ybins_centers - y_scan))  # which row
ix = np.argmin(np.abs(xbins_centers - x_scan))  # where the row starts

full_grid[iy, ix:ix+10] = ters_intensity
# Place your 10 values at the correct indices

# Mask and plot
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='grey')
ters_masked = np.ma.masked_invalid(full_grid)
xbins = np.linspace(-6.5, 6.5, 20 + 1)
ybins = np.linspace(-6.5, 6.5, 20 + 1)
plt.xticks(xbins[::2], rotation=45)
plt.yticks(ybins[::2])
plt.gca().set_xticks(xbins, minor=True)
plt.gca().set_yticks(ybins, minor=True)
plt.grid(which='minor', color='white', linewidth=0.5)
plt.imshow(ters_masked, extent=[-6.5,6.5,-6.5,6.5], origin='lower', cmap=cmap, interpolation=None)
plt.colorbar()
save_dir = Path("images")
save_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(f"{save_dir}/2d_m{args.mode}_w{width:.0f}.png", dpi=300, bbox_inches='tight')
plt.show()
