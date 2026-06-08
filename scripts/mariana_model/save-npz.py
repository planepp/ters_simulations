#!/usr/bin/env python3
"""
Save TERS2D simulation data to .npz format — all modes in one file.

Discovers every mode_NNN directory under --scan_path automatically and saves
all of them into a single .npz, mirroring the structure of
point_spectrum_generation.py so the same plotting code works for both
simulation models.

Saved arrays
------------
    atom_pos        (n_atoms, 3)          atomic positions, Angstrom
    atomic_numbers  (n_atoms,)            atomic numbers
    x_pos           (n_x,)                unique x grid coordinates, Angstrom
    y_pos           (n_y,)                unique y grid coordinates, Angstrom
    frequencies     (n_modes,)            vibrational frequencies, 1/cm
    mode_indices    (n_modes,)            original mode indices from the xyz file
    spectrums       (n_x, n_y, n_modes)   summed intensity (xx+yy+zz) per grid point per mode
    tip_height      scalar                tip-molecule distance, Angstrom

Usage
-----
    python save_ters2d_data.py geometry.xyz
    python save_ters2d_data.py geometry.xyz --scan_path ./my_scan --output data/result.npz
    python save_ters2d_data.py geometry.xyz --dq 1e-2 --efield -5e-2
"""

from pathlib import Path
import argparse
import sys
import os

import numpy as np
import ase.io

sys.path.append(os.path.expanduser("~/.local/bin"))
import finite_field_ters as ffters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_grid_coords(mode_dir: Path):
    """
    Read tip positions from all tippos_* sub-directories.

    Returns
    -------
    coords_map : dict  {tippos_int_index -> (x, y)}  in Angstrom
    """
    coords_map = {}
    for tippos_dir in sorted(mode_dir.glob("tippos_*")):
        control_file = tippos_dir / "positive_displacement" / "field_on" / "control.in"
        with open(control_file) as f:
            for line in f:
                if line.strip().startswith("rel_shift_from_tip"):
                    parts = line.split()
                    idx = int(tippos_dir.name.split("_")[1])
                    coords_map[idx] = (float(parts[1]), float(parts[2]))
                    break
    return coords_map


def read_tip_height(mode_dir: Path):
    """Return tip_molecule_distance from the first control.in found, or None."""
    for control_file in mode_dir.glob("tippos*/positive_displacement/field_on/control.in"):
        with open(control_file) as f:
            for line in f:
                if line.strip().startswith("tip_molecule_distance"):
                    return float(line.split()[-1])
    return None


def read_frequencies(xyz_path: Path):
    """Parse 'stable frequency at <value>' lines from the xyz file."""
    freqs = []
    with xyz_path.open() as f:
        for line in f:
            if "stable frequency at" in line:
                freqs.append(float(line.split()[3]))
    return freqs


def build_intensity_grid(coords_map, tippos_indices, intensity_values):
    """
    Map per-tippos intensity values onto a regular (x, y) grid.

    intensity_values has shape (n_tippos, 3) — the xx, yy, zz components of
    dadq**2. We sum over all three components to get a single scalar per tip
    position, consistent with how plot-ters2d.py uses the total intensity.

    Parameters
    ----------
    coords_map       : dict  {tippos_idx -> (x, y)}
    tippos_indices   : array-like (n_tippos,)  ordered indices matching intensity_values
    intensity_values : np.ndarray (n_tippos, 3) or (n_tippos,)

    Returns
    -------
    x_unique : np.ndarray (n_x,)
    y_unique : np.ndarray (n_y,)
    grid     : np.ndarray (n_y, n_x)   NaN where no data
    """
    intensity_values = np.array(intensity_values)
    # Sum x, y, z components -> scalar per tippos
    if intensity_values.ndim > 1:
        scalars = intensity_values.sum(axis=-1)
    else:
        scalars = intensity_values

    xs, ys, vals = [], [], []
    for idx, val in zip(tippos_indices, scalars):
        if idx in coords_map:
            x, y = coords_map[idx]
            xs.append(x)
            ys.append(y)
            vals.append(float(val))
        else:
            print(f"    Warning: tippos {idx} has no coordinate entry, skipping.")

    xs   = np.array(xs)
    ys   = np.array(ys)
    vals = np.array(vals)

    x_unique = np.unique(xs)
    y_unique = np.unique(ys)
    grid     = np.full((len(y_unique), len(x_unique)), np.nan)

    for x, y, v in zip(xs, ys, vals):
        ix = np.argmin(np.abs(x_unique - x))
        iy = np.argmin(np.abs(y_unique - y))
        grid[iy, ix] = v

    return x_unique, y_unique, grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Save all TERS2D modes to a single .npz file"
    )
    parser.add_argument("xyzfile", type=str,
                        help="Glob pattern / path to the xyz file with normal frequencies")
    parser.add_argument("--scan_path", type=str, default="./ters2d",
                        help="Path to the simulation results directory (default: ./ters2d)")
    parser.add_argument("--dq", type=float, default=5e-3,
                        help="dq used in the simulation (default: 5e-3)")
    parser.add_argument("--efield", type=float, default=-1e-1,
                        help="Electric field used in the simulation (default: -1e-1)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz path (default: ters2d_all_modes.npz)")
    args = parser.parse_args()

    working_dir = Path(args.scan_path)

    # --- Geometry ----------------------------------------------------------
    mol_system     = ase.io.read("geometry_unconstrained.in")
    periodic       = mol_system.pbc.all()
    atom_pos       = mol_system.get_positions()
    atomic_numbers = mol_system.get_atomic_numbers()

    # --- Frequencies from xyz ----------------------------------------------
    xyz_file  = next(Path(".").glob(args.xyzfile))
    all_freqs = read_frequencies(xyz_file)
    print(f"Found {len(all_freqs)} frequencies in {xyz_file.name}")

    # --- Discover all mode directories -------------------------------------
    mode_dirs = sorted(working_dir.glob("mode_[0-9][0-9][0-9]"))
    mode_list = [int(d.name.split("_")[1]) for d in mode_dirs]
    n_modes   = len(mode_list)
    print(f"Found {n_modes} mode directories: {mode_list}")

    if n_modes == 0:
        sys.exit(f"No mode_NNN directories found under {working_dir}")

    # --- Tip height (from first mode) --------------------------------------
    tip_height = read_tip_height(mode_dirs[0])
    print(f"Tip height: {tip_height} Å")

    # --- Process every mode ------------------------------------------------
    per_mode_grids = []   # (x_unique, y_unique, grid_2d) per mode
    saved_freqs    = []
    saved_indices  = []

    for m in mode_list:
        mode_dir   = working_dir / f"mode_{m:03d}"
        coords_map = read_grid_coords(mode_dir)

        print(f"  Mode {m:3d}: running ffters.analyze_2d_ters ...")
        ters = ffters.analyze_2d_ters(
            working_dir=working_dir,
            mode_idx=[m],
            efield=args.efield,
            dq=args.dq,
            periodic=periodic,
            no_groundstate=True,
        )
        # ters["intensity"] is a list with one entry per requested mode_idx.
        # We always pass one mode, so take [0] to get (n_tippos,) or (n_tippos, 3)
        intensity  = np.array(ters["intensity"][0])
        tippos_idx = np.array(ters["tippos_indices"])
        print(f"         intensity shape: {intensity.shape}, tippos count: {len(tippos_idx)}")

        x_u, y_u, grid = build_intensity_grid(coords_map, tippos_idx, intensity)
        per_mode_grids.append((x_u, y_u, grid))
        saved_freqs.append(all_freqs[m])
        saved_indices.append(m)

    # --- Save each mode's grid independently -------------------------------
    # Modes can have different grid sizes (e.g. 28 vs 196 tip positions),
    # so we cannot use a single shared 3-D array. Instead each mode is saved
    # as its own keyed entry: x_pos_NNN, y_pos_NNN, spectrum_NNN where NNN is
    # the original mode index. The plotter resolves mode number -> key via
    # mode_indices.
    output_path = Path(args.output) if args.output else Path(xyz_file.stem + ".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = dict(
        atom_pos       = atom_pos,
        atomic_numbers = atomic_numbers,
        frequencies    = np.array(saved_freqs),
        mode_indices   = np.array(saved_indices),
        tip_height     = np.float64(tip_height if tip_height is not None else np.nan),
        model          = "model_2"
    )
    for m, (x_u, y_u, grid) in zip(saved_indices, per_mode_grids):
        save_dict[f"x_pos_{m:03d}"]    = x_u
        save_dict[f"y_pos_{m:03d}"]    = y_u
        save_dict[f"spectrum_{m:03d}"] = grid.T   # (n_y, n_x) -> (n_x, n_y)

    np.savez(output_path, **save_dict)

    print(f"\nSaved -> {output_path}")
    print(f"  atom_pos       : {atom_pos.shape}")
    print(f"  atomic_numbers : {atomic_numbers.shape}")
    print(f"  frequencies    : {np.array(saved_freqs)}")
    print(f"  mode_indices   : {np.array(saved_indices)}")
    for m, (x_u, y_u, grid) in zip(saved_indices, per_mode_grids):
        print(f"  mode {m:3d}  x_pos: {x_u.shape}  y_pos: {y_u.shape}  spectrum: {grid.T.shape}")
    print(f"  tip_height     : {tip_height}")


if __name__ == "__main__":
    main()
