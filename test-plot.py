from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Grid creation ────────────────────────────────────────────────────────────

def make_edges_from_segments(segments):
    edges = []
    for i, (start, end, n) in enumerate(segments):
        e = np.linspace(start, end, n + 1)
        if i > 0:
            e = e[1:]
        edges.append(e)
    return np.concatenate(edges)

def make_edges(b, range_min=None, range_max=None):
    if np.isscalar(b):
        return np.linspace(range_min, range_max, b + 1)
    elif isinstance(b, list) and isinstance(b[0], tuple):
        return make_edges_from_segments(b)
    else:
        return np.asarray(b)

def make_test_grid(mode_dir: Path, bins: tuple):
    xedges = make_edges(bins[0])
    yedges = make_edges(bins[1])
    xcenters = 0.5 * (xedges[1:] + xedges[:-1])
    ycenters = 0.5 * (yedges[1:] + yedges[:-1])
    xx, yy = np.meshgrid(xcenters, ycenters)
    xx, yy = xx.ravel(), yy.ravel()

    mode_dir.mkdir(parents=True, exist_ok=True)

    for i_calc, (x, y) in enumerate(zip(xx, yy)):
        control_dir = mode_dir / f'tippos_{i_calc:03d}' / 'positi' / 'fieldon' / 'controlin'
        control_dir.mkdir(parents=True, exist_ok=True)
        control_file = control_dir / 'controlin'
        control_file.write_text(f'rel_shift_from_tip      {x:.6f} {y:.6f}\n')

    print(f"Created {len(xx)} tippos folders in {mode_dir}")

# ── Read coords ───────────────────────────────────────────────────────────────

def read_grid_coords(mode_dir: Path):
    coords = []
    indices = []
    for tippos_dir in sorted(mode_dir.glob('tippos_*')):
        control_file = tippos_dir / 'positi' / 'fieldon' / 'controlin' / 'controlin'
        if not control_file.exists():
            continue
        idx = int(tippos_dir.name.split('_')[1])
        with open(control_file) as f:
            for line in f:
                if line.strip().startswith('rel_shift_from_tip'):
                    parts = line.split()
                    coords.append((float(parts[1]), float(parts[2])))
                    indices.append(idx)
                    break
    return np.array(coords), np.array(indices)

def get_coord(coords, indices, tippos_idx):
    i = np.where(indices == tippos_idx)[0][0]
    return coords[i]

# ── Edges ─────────────────────────────────────────────────────────────────────

def compute_edges(centers):
    centers = np.sort(centers)
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    inner = 0.5 * (centers[:-1] + centers[1:])
    left  = centers[0]  - (centers[1]  - centers[0])  / 2
    right = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.concatenate([[left], inner, [right]])

# ── Fake intensity ────────────────────────────────────────────────────────────

def fake_intensity(coords, indices):
    def gaussian(x, y, x0, y0, sigma):
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    intensity_per_tippos = {}
    for tippos_idx, (x, y) in zip(indices, coords):
        val = (  gaussian(x, y, x0=-2, y0=-2, sigma=2.0)
               + gaussian(x, y, x0= 3, y0= 3, sigma=1.0) * 0.6
               + np.random.normal(0, 0.02))
        intensity_per_tippos[tippos_idx] = val
    return intensity_per_tippos

# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_ters(coords, indices, intensity_per_tippos, title='TERS image'):
    x_centers = np.unique(coords[:, 0])
    y_centers = np.unique(coords[:, 1])
    xedges = compute_edges(x_centers)
    yedges = compute_edges(y_centers)

    grid = np.full((len(y_centers), len(x_centers)), np.nan)
    for tippos_idx, intensity in intensity_per_tippos.items():
        i = np.where(indices == tippos_idx)[0]
        if len(i) == 0:
            continue
        x, y = coords[i[0]]
        ix = np.argmin(np.abs(x_centers - x))
        iy = np.argmin(np.abs(y_centers - y))
        grid[iy, ix] = intensity

    masked = np.ma.masked_invalid(grid)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='grey')

    fig, ax = plt.subplots()
    im = ax.pcolormesh(xedges, yedges, masked, cmap=cmap)
    ax.scatter(coords[:, 0], coords[:, 1], s=5, color='white', alpha=0.3, zorder=3)
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mode_dir = Path('test/mode_000')

    bins = (
        [(-10, -3, 1), (-3, 3, 20), (3, 10, 1)],
        [(-10, -3, 1), (-3, 3, 20), (3, 10, 1)],
    )

    make_test_grid(mode_dir, bins)
    coords, indices = read_grid_coords(mode_dir)
    intensity_per_tippos = fake_intensity(coords, indices)
    plot_ters(coords, indices, intensity_per_tippos, title='TERS test')
