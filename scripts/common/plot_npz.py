import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from ase.data.colors import jmol_colors
from ase.data import covalent_radii
from scipy.interpolate import RegularGridInterpolator

parser = argparse.ArgumentParser(description="Calculate and plot a 2D TERS image")
parser.add_argument("npzfile", type=str, help="Path to the .npz file with TERS data")
parser.add_argument("mode", type=int, help="Primary mode index to plot (original mode number)")
parser.add_argument("--modes", type=int, nargs='+', default=None, help="Additional mode indices to sum over. The primary mode is always included.")
parser.add_argument("--molecule", action='store_true', help="Overlay molecule on top of the TERS map")
parser.add_argument("--interpolate", action='store_true', help="Interpolate between grid positions")
parser.add_argument("--savepng", action='store_true', help="Save the figure as a PNG file")
parser.add_argument("--rotate", type=float, default=0.0, help="Rotate the TERS map and molecule by this angle in degrees (counterclockwise)")
args = parser.parse_args()

# --------------------
# Load data
# --------------------
data         = np.load(args.npzfile)
freqs        = data["frequencies"]
mode_indices = data["mode_indices"]
positions    = data["atom_pos"]
numbers      = data["atomic_numbers"]
model        = data["model"]
print(f"This data was saved with {model}")

all_modes = [args.mode] + (args.modes if args.modes else [])

# --------------------
# Report available modes
# --------------------
print(f"Available modes in {Path(args.npzfile).name}:")
for m, f in zip(mode_indices, freqs):
    print(f"  Mode {m:3d}: {f:.1f} cm⁻¹")
print()

# Validate
available = set(mode_indices.tolist())
for m in all_modes:
    if m not in available:
        raise ValueError(f"Mode {m} not found in {args.npzfile}. Available: {sorted(available)}")

def get_grid(m):
    return data[f"x_pos_{m:03d}"], data[f"y_pos_{m:03d}"], data[f"spectrum_{m:03d}"]

def freq_of(m):
    return freqs[np.where(mode_indices == m)[0][0]]

# --------------------
# Load grids and sum
# --------------------
x, y, z = get_grid(args.mode)
z = z.copy()

for m in (args.modes or []):
    xm, ym, zm = get_grid(m)
    if xm.shape == x.shape and ym.shape == y.shape and np.allclose(xm, x) and np.allclose(ym, y):
        z += zm
    else:
        X, Y = np.meshgrid(x, y, indexing='ij')
        interp_fn = RegularGridInterpolator(
            (xm, ym), zm, method='linear', bounds_error=False, fill_value=0.0
        )
        z += interp_fn((X, Y))

# --------------------
# Sanity check
# --------------------
print(f"x_pos : {x.shape},  range [{x.min():.2f}, {x.max():.2f}] Å")
print(f"y_pos : {y.shape},  range [{y.min():.2f}, {y.max():.2f}] Å")
print(f"Summing : {all_modes}")
for m in all_modes:
    print(f"  Mode {m}: {freq_of(m):.3f} cm⁻¹")

# --------------------
# Rotate grid and molecule
# --------------------
if args.rotate != 0.0:
    angle_rad = np.deg2rad(args.rotate)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])

    # Rotate molecule positions (x, y only)
    positions = positions.copy()
    positions[:, :2] = (R @ positions[:, :2].T).T

    # Find new grid extent from rotated corners
    corners = np.array([
        [x.min(), y.min()],
        [x.max(), y.min()],
        [x.min(), y.max()],
        [x.max(), y.max()],
    ])
    rotated_corners = (R @ corners.T).T
    x_min, y_min = rotated_corners.min(axis=0)
    x_max, y_max = rotated_corners.max(axis=0)

    # New regular grid with same number of points
    x_new = np.linspace(x_min, x_max, len(x))
    y_new = np.linspace(y_min, y_max, len(y))

    # Back-map new grid points into original frame and interpolate
    X_new, Y_new = np.meshgrid(x_new, y_new, indexing='ij')
    coords_orig = (R.T @ np.stack([X_new.ravel(), Y_new.ravel()])).T
    interp_fn = RegularGridInterpolator(
        (x, y), z, method='linear', bounds_error=False, fill_value=0.0
    )
    z = interp_fn(coords_orig).reshape(X_new.shape)
    x, y = x_new, y_new

# --------------------
# Build figure
# --------------------
fig, ax = plt.subplots(figsize=(6, 5))
interp_method = "bilinear" if args.interpolate else "nearest"

im = ax.imshow(
    z,
    cmap="viridis",
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
    aspect="auto",
    interpolation=interp_method
)
#plt.colorbar(im, ax=ax, label="TERS intensity")

if args.molecule:
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=[jmol_colors[n] for n in numbers],
        s=[covalent_radii[n] * 100 for n in numbers],
        edgecolors='k',
        linewidths=0.4,
        zorder=3
    )

mode_str = "+".join(str(m) for m in all_modes)
freq_str = ", ".join(f"{freq_of(m):.1f}" for m in all_modes)
ax.set_title(rf'TERS image, modes {mode_str} ({freq_str} 1/cm)')
ax.set_xlabel(r'$x$ [$\mathrm{\AA}$]')
ax.set_ylabel(r'$y$ [$\mathrm{\AA}$]')
plt.tight_layout()

if args.savepng:
    save_dir = Path("images")
    save_dir.mkdir(parents=True, exist_ok=True)
    outfile = save_dir / f"2d_m{mode_str}.png"
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {outfile}")

plt.show()
