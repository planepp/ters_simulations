import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

parser = argparse.ArgumentParser(description="Plot a 1D TERS spectrum at a given (x, y) tip position")
parser.add_argument("npzfile", type=str, help="Path to the .npz file with TERS data")
parser.add_argument("x", type=float, help="Tip x position in Å")
parser.add_argument("y", type=float, help="Tip y position in Å")
parser.add_argument("--modes", type=int, nargs='+', default=None,
                    help="Subset of mode indices to include (default: all available modes)")
parser.add_argument("--broaden", type=float, default=None,
                    help="Lorentzian broadening FWHM in cm⁻¹ (default: stick spectrum)")
parser.add_argument("--freqrange", type=float, nargs=2, default=None,
                    metavar=('FMIN', 'FMAX'),
                    help="Frequency range to plot in cm⁻¹, e.g. --freqrange 200 1800")
parser.add_argument("--interpolate", action='store_true',
                    help="Bilinear-interpolate intensity to exact (x, y); "
                         "default snaps to nearest grid point")
parser.add_argument("--savepng", action='store_true', help="Save the figure as a PNG file")
args = parser.parse_args()

# --------------------
# Load data
# --------------------
data         = np.load(args.npzfile)
freqs        = data["frequencies"]
mode_indices = data["mode_indices"]
model        = data["model"]
print(f"This data was saved with {model}")

# --------------------
# Report available modes
# --------------------
print(f"Available modes in {Path(args.npzfile).name}:")
for m, f in zip(mode_indices, freqs):
    print(f"  Mode {m:3d}: {f:.1f} cm⁻¹")
print()

# Filter to requested subset
if args.modes is not None:
    available = set(mode_indices.tolist())
    for m in args.modes:
        if m not in available:
            raise ValueError(f"Mode {m} not found. Available: {sorted(available)}")
    mask = np.isin(mode_indices, args.modes)
else:
    mask = np.ones(len(mode_indices), dtype=bool)

sel_modes = mode_indices[mask]
sel_freqs = freqs[mask]

# --------------------
# Extract intensity at (x, y) for every selected mode
# --------------------
intensities = np.zeros(len(sel_modes))

for i, m in enumerate(sel_modes):
    x_grid = data[f"x_pos_{m:03d}"]
    y_grid = data[f"y_pos_{m:03d}"]
    z_grid = data[f"spectrum_{m:03d}"]   # shape (len(x_grid), len(y_grid))

    if args.interpolate:
        interp_fn = RegularGridInterpolator(
            (x_grid, y_grid), z_grid,
            method='linear', bounds_error=False, fill_value=0.0
        )
        intensities[i] = float(interp_fn([[args.x, args.y]]))
    else:
        ix = np.argmin(np.abs(x_grid - args.x))
        iy = np.argmin(np.abs(y_grid - args.y))
        intensities[i] = z_grid[ix, iy]

# --------------------
# Sanity report
# --------------------
snap_x = data[f"x_pos_{sel_modes[0]:03d}"]
snap_y = data[f"y_pos_{sel_modes[0]:03d}"]
actual_x = snap_x[np.argmin(np.abs(snap_x - args.x))]
actual_y = snap_y[np.argmin(np.abs(snap_y - args.y))]
if args.interpolate:
    print(f"Interpolated at (x, y) = ({args.x:.3f}, {args.y:.3f}) Å")
else:
    print(f"Requested (x, y) = ({args.x:.3f}, {args.y:.3f}) Å  →  "
          f"snapped to ({actual_x:.3f}, {actual_y:.3f}) Å")
print(f"Modes included : {len(sel_modes)}")
print(f"Frequency range: [{sel_freqs.min():.1f}, {sel_freqs.max():.1f}] cm⁻¹")

# --------------------
# Optional broadening
# --------------------
plot_freq_range = args.freqrange if args.freqrange else [sel_freqs.min() - 50, sel_freqs.max() + 50]

if args.broaden is not None:
    freq_axis = np.linspace(plot_freq_range[0], plot_freq_range[1], 2000)
    gamma = args.broaden / 2.0  # half-width at half-maximum
    spectrum = np.zeros_like(freq_axis)
    for f, I in zip(sel_freqs, intensities):
        spectrum += I * (gamma**2) / ((freq_axis - f)**2 + gamma**2)
else:
    freq_axis = sel_freqs
    spectrum  = intensities

# --------------------
# Build figure
# --------------------
fig, ax = plt.subplots(figsize=(8, 4))

if args.broaden is not None:
    ax.plot(freq_axis, spectrum, color='steelblue', lw=1.5)
    ax.fill_between(freq_axis, spectrum, alpha=0.25, color='steelblue')
else:
    # Stick spectrum
    ax.vlines(freq_axis, 0, spectrum, color='steelblue', lw=1.2)
    ax.scatter(freq_axis, spectrum, s=20, color='steelblue', zorder=3)

if args.freqrange:
    ax.set_xlim(args.freqrange)

ax.set_xlabel(r'Frequency [cm$^{-1}$]')
ax.set_ylabel('TERS intensity')

pos_label = f"({args.x:.2f}, {args.y:.2f}) Å"
if not args.interpolate:
    pos_label += f"  →  snapped to ({actual_x:.2f}, {actual_y:.2f}) Å"
ax.set_title(rf'TERS spectrum at {pos_label}')

plt.tight_layout()

if args.savepng:
    save_dir = Path("images")
    save_dir.mkdir(parents=True, exist_ok=True)
    outfile = save_dir / f"1d_x{args.x:.2f}_y{args.y:.2f}.png"
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {outfile}")

plt.show()
