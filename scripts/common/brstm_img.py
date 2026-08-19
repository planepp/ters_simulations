#!/usr/bin/env python3
"""
Plot constant-height / constant-current AFM, STM, and BRSTM images from
simulation .npz output. Rebuilt from brstm.ipynb.

Run this script from the directory that contains `input.in` (with `LABEL = ...`
and `ZREF = ...` lines), `grid.npz`, `sample/POSCAR`, the `relax_*.npz` AFM
file, and the `stm_*.npz` files, e.g.:

    cd /scratch/project_2001912/plane/cobr_shigeki/phase1/moleculeonslab/rndm
    python brstm_img.py --modes afm stm --afm-zs 3.0 3.2 3.4 --zs 2.7

See `python brstm_img.py --help` for all options.
"""

import argparse
import os
import re
from argparse import Namespace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

import tricubic
from ase.io import read
from ase.data import covalent_radii
from ase.data.colors import jmol_colors


# Fixed AFM force-field parameters (not exposed as CLI args).
AFM_ALPHA = 1.08
AFM_V0 = 42.91
AFM_KAPPA = 0.4


# --------------------------------------------------------------------------
# Core math
# --------------------------------------------------------------------------

def constant_height(ldos, z, z_range, zo, cell, s=None, repeat=(1, 1)):
    """Extract a constant-height slice at height `z` from a 3D LDOS/force cube."""
    nz = ldos.shape[2]
    ldos_ = ldos.reshape((-1, nz))
    I = np.empty(ldos_.shape[0])

    dz = z_range / nz
    zp = round((z - zo) / dz)

    for i, a in enumerate(ldos_):
        I[i] = a[zp]

    s0 = I.shape = ldos.shape[:2]
    if s:
        I = gaussian_filter(I, sigma=s, mode='wrap')
    I = np.tile(I, repeat)
    s = I.shape

    ij = np.indices(s, dtype=float).reshape((2, -1)).T
    x, y = np.dot(ij / s0, cell[:2, :2]).T.reshape((2,) + s)

    return x, y, I


def constant_current(I, Z, iso_val, z_range, cell):
    """Find the z at which `I` crosses `iso_val` along the z-axis, per (x, y)."""
    s0 = I.shape[:2]
    ij = np.indices(s0, dtype=float).reshape((2, -1)).T
    x, y = np.dot(ij / s0, cell[:2, :2]).T.reshape((2,) + s0)

    idx = np.where((Z >= z_range[0]) & (Z <= z_range[1]))[0]
    if idx.size == 0:
        return x, y, np.full(x.shape, np.nan)
    search_range = (idx.min(), idx.max())

    def interp_row(Ixy, Z, iso_val, search_range):
        sub_row = Ixy[search_range[0]:search_range[1] + 1]
        sub_sign = np.sign(sub_row - iso_val)
        diff_sign = np.diff(sub_sign)
        crossing_indices = np.where(diff_sign != 0)[0]
        if crossing_indices.size == 0:
            return z_range[0]
        i = crossing_indices[0] + search_range[0]
        if i >= len(Ixy) - 1:
            return np.nan
        f = interp1d([Ixy[i], Ixy[i + 1]], [Z[i], Z[i + 1]])
        return f(iso_val)

    surface_z_flat = np.apply_along_axis(
        lambda Ixy: interp_row(Ixy, Z, iso_val, search_range),
        axis=2,
        arr=I,
    )
    surface_z = surface_z_flat.reshape(x.shape)
    return x, y, surface_z


# --------------------------------------------------------------------------
# Input parsing
# --------------------------------------------------------------------------

def read_input_in(wd):
    """Read `LABEL = ...` and `ZREF = ...` out of input.in in the working directory."""
    input_file = wd / 'input.in'
    if not input_file.exists():
        raise FileNotFoundError(f"Could not find input.in in {wd}")

    text = input_file.read_text()

    label_match = re.search(r'^\s*LABEL\s*=\s*(\S+)', text, re.IGNORECASE | re.MULTILINE)
    if not label_match:
        raise ValueError(f"Could not find a 'LABEL = ...' line in {input_file}")

    zref_match = re.search(r'^\s*ZREF\s*=\s*(\S+)', text, re.IGNORECASE | re.MULTILINE)
    if not zref_match:
        raise ValueError(f"Could not find a 'ZREF = ...' line in {input_file}")

    return label_match.group(1), float(zref_match.group(1))


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot constant-height / constant-current AFM, STM, and BRSTM images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--wd', type=Path, default=None,
                    help="Working directory (default: directory the script is run from).")
    p.add_argument('--modes', nargs='+', choices=['afm', 'stm', 'brstm'], default=['stm'],
                    help="Which image type(s) to generate.")
    p.add_argument('--no-show', dest='show', action='store_false', default=True,
                    help="Don't call plt.show(); only save files.")
    p.add_argument('--repeat', nargs=2, type=int, default=[1, 1],
                    help="Tile the plotted cell (nx ny). Applies to STM and BRSTM "
                         "constant-height images.")

    # AFM (heights only -- no bias/weight concept)
    g_afm = p.add_argument_group('afm')
    g_afm.add_argument('--afm-zs', nargs='+', type=float, default=[3.2],
                        help="Tip heights (A) for AFM constant-height force images.")

    # Shared STM / BRSTM panel parameters -- both modes use the exact same
    # panels (same heights/iso-values, biases, and weights) so images are
    # directly comparable.
    g_panels = p.add_argument_group('stm / brstm panels')
    g_panels.add_argument('--plot-type', choices=['height', 'current'], default='height',
                           help="Constant-height or constant-current images.")
    g_panels.add_argument('--zs', nargs='+', type=float, default=[3.2],
                           help="Tip heights (A) above zref, for constant-height images.")
    g_panels.add_argument('--iso-vals', nargs='+', type=float, default=[1e-7],
                           help="Current iso-values, for constant-current images.")
    g_panels.add_argument('--biases', nargs='+', type=float, default=[0.5],
                           help="Bias voltage(s) (V), one per panel (or a single value "
                                "broadcast to all panels).")
    g_panels.add_argument('--weights', nargs='+', type=float, default=[1.0],
                           help="s-orbital weight (0-1) per panel; p-weight = 1 - s.")

    return p.parse_args()


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_grid_and_atoms(wd):
    sample = Namespace()
    sample.grid = Namespace(**np.load(wd / 'grid.npz'))
    at = read(wd / 'sample/POSCAR')
    return sample, at


def load_afm_data(afm_path, label):
    fname = afm_path / f'relax_{label}_k{AFM_KAPPA:.4f}_a{AFM_ALPHA:.2f}_V{AFM_V0:.2f}.npz'
    if not fname.exists():
        raise FileNotFoundError(f"AFM data not found: {fname}")
    afm = Namespace(**np.load(fname))
    afm.F = np.gradient(afm.E, -afm.dr[2], axis=2)
    return afm


def load_stm_data(sample, stm_path, label):
    """Scan stm_path for stm_*V*.npz files, load them, and build the z-axis."""
    sample.stm = dict()
    npz_files = [f for f in os.listdir(stm_path) if f.endswith('.npz')]

    all_biases = []
    for file in npz_files:
        match = re.search(r"(?=.*stm).*V([-\d.]+)", file, re.IGNORECASE)
        if match:
            bias = match.group(1).rstrip('.')
            all_biases.append(float(bias))

    if not all_biases:
        raise FileNotFoundError(f"No stm_*V*.npz files found in {stm_path}")

    V = np.array(sorted(set(all_biases)), dtype=float)
    for v in V:
        sample.stm[f'{v:.3f}'] = Namespace(**np.load(stm_path / f'stm_{label}_V{v:.3f}.npz'))

    sample.stm['z'] = np.linspace(
        0, sample.grid.cell[2, 2],
        sample.stm[f'{V[0]:.3f}'].s.shape[2],
        endpoint=False,
    )
    return sample, V


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _broadcast(values, n):
    """Repeat a single-item list to length n; otherwise return it unchanged."""
    return values * n if len(values) == 1 else values


def _require_matching_lengths(*named_lists):
    lengths = {name: len(vals) for name, vals in named_lists}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Argument lists must be the same length, got: {lengths}")


def _panel_params(args):
    """Broadcast weights/biases/zs-or-iso-vals to a common panel count.

    Shared by STM and BRSTM so both use identical panels.
    """
    weights = args.weights
    n = len(weights)
    vals = args.zs if args.plot_type == 'height' else args.iso_vals
    vals = _broadcast(vals, n)
    biases = _broadcast(args.biases, n)
    _require_matching_lengths(('weights', weights), ('zs/iso-vals', vals), ('biases', biases))
    return vals, biases, weights


def _save_and_show(fig, out, args):
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f'Saved {out}')
    if args.show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_afm(afm, args, img_path, cell, zref):
    zmin_afm = afm.z.min()
    zmax_afm = afm.z.max()
    z_range_afm = zmax_afm - zmin_afm

    zs = args.afm_zs
    n = len(zs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for i, z in enumerate(zs):
        ax = axes[0, i]
        ax.contourf(
            *constant_height(afm.F, z, z_range_afm, zmin_afm - zref, cell, s=0),
            cmap='gray',
            levels=256,
        )
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(f'{z}' + r' $\mathring{A}$', fontsize=18)

    out = img_path / f'afm_z{zs}.png'
    _save_and_show(fig, out, args)


def plot_stm(sample, args, img_path, cell, zref):
    vals, biases, weights = _panel_params(args)
    n = len(weights)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    if args.plot_type == 'height':
        for i, (z, bias, weight_s) in enumerate(zip(vals, biases, weights)):
            ax = axes[0, i]
            weight_p = 1 - weight_s
            I = sample.stm[f'{bias:.3f}']
            ax.contourf(
                *constant_height(
                    weight_s * I.s + gaussian_filter(weight_p * I.p, sigma=2, mode='wrap'),
                    z + zref,
                    sample.grid.cell[2, 2],
                    0,
                    cell,
                    s=0,
                    repeat=tuple(args.repeat),
                ),
                cmap='gray',
                levels=256,
            )
            ax.set_aspect('equal')
            ax.set_axis_off()
            ax.set_title(f'{z}' + r' $\mathring{A}$ , ' + f'{bias} V , ' + f'{weight_s * 100:.0f} % s',
                          fontsize=18)
        out = img_path / f'stm_height_z{vals}_V{biases}_s{weights}.png'

    else:  # current
        Z = sample.stm['z']
        z_range = [Z.min(), Z.max()]
        for i, (iso_val, bias, weight_s) in enumerate(zip(vals, biases, weights)):
            ax = axes[0, i]
            weight_p = 1 - weight_s
            I = sample.stm[f'{bias:.3f}']
            cs = ax.contourf(
                *constant_current(
                    weight_s * I.s + gaussian_filter(weight_p * I.p, sigma=2, mode='wrap'),
                    Z, iso_val, z_range, cell,
                ),
                cmap='gray',
                levels=256,
            )
            fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
            ax.set_aspect('equal')
            ax.set_axis_off()
            ax.set_title(f'{iso_val} A , {bias} V , {weight_s * 100:.0f}% s', fontsize=18)
        out = img_path / f'stm_current_iso{vals}_V{biases}_s{weights}.png'

    _save_and_show(fig, out, args)


def _compute_brstm_interpolation(sample, afm, at, bias):
    """Interpolate the STM LDOS onto the relaxed AFM tip trajectory, for one bias."""
    zmin_afm = afm.z.min()
    tip_pos = np.moveaxis(afm.tip, 0, -1).copy()
    tip_pos[..., 2] += 3.02
    shape = afm.F.shape
    cell_ = at.cell.copy()
    cell_[2, 2] = shape[2] * afm.dr[2]

    data = sample.stm[f'{bias:.3f}']
    ldos_shape = data.s.shape

    ijk = np.indices(shape, dtype=float).reshape((3, -1)).T
    X, Y, Z = np.dot(ijk / shape, cell_).T.reshape((3,) + shape)
    X += tip_pos[..., 0]
    Y += tip_pos[..., 1]
    Z += tip_pos[..., 2] + zmin_afm

    rlx = {}
    for orb, weight_s in zip(['s', 'p'], [1, 0]):
        weight_p = 1 - weight_s
        interp = tricubic.tricubic(
            list(weight_s * data.s + gaussian_filter(weight_p * data.p, sigma=2, mode='wrap')),
            list(data.s.shape),
        )
        res = np.zeros(tip_pos.shape[:-1])
        it = np.nditer([X, Y, Z], flags=['multi_index'])
        for xi, yi, zi in it:
            res[it.multi_index] = interp.ip(
                list(np.linalg.solve(at.cell.T, [xi, yi, zi]) * ldos_shape)
            )
        rlx[orb] = res.copy()

    return rlx, cell_


def plot_brstm(sample, afm, at, args, img_path, zref):
    zmin_afm = afm.z.min()
    zmax_afm = afm.z.max()
    z_range_afm = zmax_afm - zmin_afm

    vals, biases, weights = _panel_params(args)
    n = len(weights)

    # Interpolation is done once per unique bias present in the panels.
    rlx_cache = {}
    cell_ = None
    for bias in set(biases):
        rlx_cache[bias], cell_ = _compute_brstm_interpolation(sample, afm, at, bias)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    if args.plot_type == 'height':
        for i, (z, bias, weight_s) in enumerate(zip(vals, biases, weights)):
            ax = axes[0, i]
            weight_p = 1 - weight_s
            rlx = rlx_cache[bias]
            data_sp = weight_s * rlx['s'] + weight_p * rlx['p']
            ax.contourf(
                *constant_height(
                    data_sp, z, z_range_afm, zmin_afm - zref, cell_, s=0,
                    repeat=tuple(args.repeat),
                ),
                cmap='gray',
                levels=256,
            )
            ax.set_aspect('equal')
            ax.set_axis_off()
            ax.set_title(f'{z}' + r' $\mathring{A}$ , ' + f'{bias} V , ' + f'{weight_s * 100:.0f} % s',
                          fontsize=18)
        out = img_path / f'brstm_height_z{vals}_V{biases}_s{weight_s}.png'

    else:  # current
        Z = sample.stm['z']
        z_range = [zmin_afm - zref, zmax_afm - zref]
        for i, (iso_val, bias, weight_s) in enumerate(zip(vals, biases, weights)):
            ax = axes[0, i]
            weight_p = 1 - weight_s
            rlx = rlx_cache[bias]
            data_sp = weight_s * rlx['s'] + weight_p * rlx['p']
            cs = ax.contourf(
                *constant_current(data_sp, Z, iso_val, z_range, cell_),
                cmap='gray',
                levels=256,
            )
            fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
            ax.set_aspect('equal')
            ax.set_axis_off()
            ax.set_title(f'{iso_val} A , {bias} V , {weight_s * 100:.0f}% s', fontsize=18)
        out = img_path / f'brstm_current_iso{vals}_V{biases}_s{weight_s}.png'

    _save_and_show(fig, out, args)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()

    wd = args.wd if args.wd is not None else Path.cwd()
    label, zref = read_input_in(wd)
    print(f'Working directory: {wd}')
    print(f'Label: {label}')
    print(f'zref: {zref}')

    afm_path = wd
    stm_path = wd
    image_path = wd / 'images'

    sample, at = load_grid_and_atoms(wd)

    need_afm = 'afm' in args.modes or 'brstm' in args.modes
    need_stm = 'stm' in args.modes or 'brstm' in args.modes

    afm = load_afm_data(afm_path, label) if need_afm else None
    if need_stm:
        sample, _ = load_stm_data(sample, stm_path, label)

    if 'afm' in args.modes:
        img_path_afm = image_path / 'afm'
        img_path_afm.mkdir(parents=True, exist_ok=True)
        plot_afm(afm, args, img_path_afm, sample.grid.cell, zref)

    if 'stm' in args.modes:
        img_path_stm = image_path / 'stm'
        img_path_stm.mkdir(parents=True, exist_ok=True)
        plot_stm(sample, args, img_path_stm, sample.grid.cell, zref)

    if 'brstm' in args.modes:
        img_path_brstm = image_path / 'brstm'
        img_path_brstm.mkdir(parents=True, exist_ok=True)
        plot_brstm(sample, afm, at, args, img_path_brstm, zref)


if __name__ == '__main__':
    main()
