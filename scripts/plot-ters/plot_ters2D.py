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
parser.add_argument("mode", type=str, default=None, help="Mode index to process: single integer")
parser.add_argument("--wth", type=float, default=0, help="Width in 1/cm over which the nearby modes should be found and summed over.")
parser.add_argument("--rot", type=float, default=0, help="Angle by which the image should be rotated.")
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
            elif line.startswith("xx"):
                xx = float (line.split('=')[-1].strip().rstrip(','))
            elif line.startswith("yy"):
                yy = float (line.split('=')[-1].strip().rstrip(','))
else:
    print("run-ters.py not found in working directory. Assuming default dq, efield.")
    print("run-ters.py not found in working directory. Please add this file with the nbins = (n,m) and xx, yy tags.")
    exit(1)

print(f"dq = {dq}")
print(f"efield = {efield}")
print(f"nbins = {nbins}")
print(f"xx = {xx}")
print(f"yy = {yy}")

ex = np.array([-xx, xx, -yy, yy])
unconstrained_geometry_file = "geometry_unconstrained.in"
mol_system = ase.io.read(Path(unconstrained_geometry_file))
periodic = mol_system.pbc.all()
positions = mol_system.get_positions()
numbers = mol_system.get_atomic_numbers()

mode_idx = args.mode
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

# Save raw data
if mode_idx is None:
    # print all frequencies
    for i, fval in enumerate(freqs):
        print(f"Mode {i}: {fval:.3f} 1/cm")
else:
    # mode_idx is a list of integers
    for idx in mode_idx:
        if 0 <= idx < len(freqs):  # use < len(freqs) to avoid IndexError
            f = freqs[idx]
            print(f"Mode {idx}: {f:.3f} 1/cm")
        else:
            print(f"Invalid mode_idx {idx}, file has {len(freqs)} modes")

ters = ffters.analyze_2d_ters(working_dir=Path('./ters2d'), mode_idx = mode_idx, efield=efield, dq=dq, nbins=nbins, periodic=periodic, no_groundstate=True)

# save the raw intensity data
for mode in mode_idx:
    np.savetxt(f"rawdata/intensity_{freqs[mode]}.dat", ters['intensity'])

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
ters_intensity = read_and_sum_intensities(files)

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
xmin, xmax = -xx, yy
ymin, ymax = -yy, yy

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
plt.imshow(ters_intensity, origin='lower', extent=ex, cmap='viridis', interpolation='bilinear')
plt.xlabel(r'$x$ [$\mathrm{\AA}$]')
plt.ylabel(r'$y$ [$\mathrm{\AA}$]')
if '-' not in args.mode and ',' not in args.mode:
    plt.title(rf'TERS image d={tip_height} $\mathrm{{\AA}}$, f = {selected_freqs[0]:.3f} 1/cm ({min(new_mods):.0f}-{max(new_mods):.0f})')
else:
    plt.title(rf'TERS image d={tip_height} $\mathrm{{\AA}}$, {args.mode} modes')

xbins = np.linspace(-xx, xx, nbins[0] + 1)
ybins = np.linspace(-yy, yy, nbins[1] + 1)
plt.xticks(xbins[::2], rotation=45)
plt.yticks(ybins[::2])
plt.gca().set_xticks(xbins, minor=True)
plt.gca().set_yticks(ybins, minor=True)

plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])

plt.grid(which='minor', color='white', linewidth=0.5)
plt.tight_layout()

save_dir = Path("images")
save_dir.mkdir(parents=True, exist_ok=True) 
plt.savefig(f"{save_dir}/2d_m{args.mode}_w{width:.0f}.png", dpi=300, bbox_inches='tight')
plt.show()
