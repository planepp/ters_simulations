import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

from ase.data.colors import jmol_colors
from ase.data import covalent_radii

from scipy.interpolate import RegularGridInterpolator


# --------------------
# args
# --------------------
parser = argparse.ArgumentParser()

parser.add_argument("npzfile", type=str, nargs="?", default=None)
parser.add_argument("mode", type=int, nargs="?", default=None)
parser.add_argument("--modes", type=int, nargs='+', default=None)
parser.add_argument("--molecule", action="store_true")
parser.add_argument("--interpolate", action="store_true")
parser.add_argument("--savepng", action="store_true")

args = parser.parse_args()

# --------------------
# npzfile: ask if missing
# --------------------
if args.npzfile is None:
    print("No npzfile provided. Usage: plot_npz.py <npzfile> <mode> [options]")
    sys.exit(1)

# --------------------
# load data
# --------------------
data = np.load(args.npzfile)
print("Model:", data['model'])
print("Mode indices available:", data['mode_indices'])

freqs = data["frequencies"]
mode_indices = data["mode_indices"]
positions = data["atom_pos"]
numbers = data["atomic_numbers"]

# --------------------
# mode: check missing / invalid
# --------------------
if args.mode is None or args.mode not in mode_indices:
    print("No valid mode provided. Please choose a mode from the list above.")
    sys.exit(1)

all_modes = [args.mode] + (args.modes or [])

# --------------------
# mode: ask if missing
# --------------------
if args.mode is None or args.mode not in mode_indices:
    print("No mode provided. Please enter a mode in the mode available list above. \n")

def get_grid(m):
    return (
        data[f"x_pos_{m:03d}"],
        data[f"y_pos_{m:03d}"],
        data[f"spectrum_{m:03d}"]
    )


def freq_of(m):
    return freqs[np.where(mode_indices == m)[0][0]]


# --------------------
# sum modes
# --------------------
x, y, z = get_grid(args.mode)
z = z.copy()

for m in (args.modes or []):
    xm, ym, zm = get_grid(m)

    interp = RegularGridInterpolator(
        (xm, ym),
        zm,
        method="linear",
        bounds_error=False,
        fill_value=0.0
    )

    X, Y = np.meshgrid(x, y, indexing="ij")
    z += interp((X, Y))


# --------------------
# extent
# --------------------
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()


# --------------------
# plot
# --------------------
fig, ax = plt.subplots(figsize=(6, 5))
if data['model'] == 'simple model':
    z = z.T  # rotate x,y axis with Mariana model (double transposing)

# for plotting 1D arrays: pad the singular axis so imshow doesn't get a zero-range extent
if ymin == ymax:
    ymin, ymax = ymin - 0.5, ymax + 0.5
    z = np.tile(z[:, 0:1], (1, 2))
elif xmin == xmax:
    xmin, xmax = xmin - 0.5, xmax + 0.5
    z = np.tile(z[0:1, :], (2, 1))

im = ax.imshow(
    z.T,
    cmap="viridis",
    origin="lower",
    extent=[xmin, xmax, ymin, ymax],
    aspect="auto",
    interpolation="bilinear" if args.interpolate else "nearest"
)
fig.colorbar(im, ax=ax)


# --------------------
# molecule overlay
# --------------------
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


# --------------------
# labels
# --------------------
mode_str = "+".join(str(m) for m in all_modes)
freq_str = ", ".join(f"{freq_of(m):.1f}" for m in all_modes)

ax.set_title(f"TERS image modes {mode_str} ({freq_str} cm⁻¹)")
ax.set_xlabel("x [Å]")
ax.set_ylabel("y [Å]")


plt.tight_layout()


# --------------------
# save
# --------------------
if args.savepng:
    outdir = Path("images")
    outdir.mkdir(exist_ok=True, parents=True)

    outfile = outdir / f"TERS_modes_{mode_str}.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Saved to {outfile}")


plt.show()
