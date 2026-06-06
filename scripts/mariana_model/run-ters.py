#!/usr/bin/env python
from pathlib import Path
import numpy as np
import sys
import os
import shutil
import ast
from ase.io import read, write

sys.path.append(os.path.expanduser("~/.local/bin"))

from finite_field_ters import FiniteFieldTERS
from prepare_ters import (
    h, masses, geo_unconstrained, geo_constrained,
    species_dir, symbols_constrained, symbols_unconstrained,
    find_species_file
)

def zero_homogeneous_field(geom_path):
    with open(geom_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "homogeneous_field" in line:
            parts = line.split()
            lines[i] = f"{parts[0]} 0 0 0\n"
    with open(geom_path, "w") as f:
        f.writelines(lines)


storage_dir = Path('~') # Triton
# storage_dir = Path('/projappl/project_2001912') # CSC

ters = FiniteFieldTERS(
hessian = h,
modes = None,
masses = masses,
dq = 5e-3,
efield = -1e-1,
storage_dir = storage_dir,
fn_control_template = Path.cwd() / 'control.in',
species_dir = Path(species_dir),
#fn_tip_groundstate = Path('zeros.cube'),
fn_tip_groundstate = None,
fn_tip_derivative = Path('tipA_05_vh_ft_0049_3221meV_x1000.cube'),
#fn_elsi_restart = Path('D_spin_01_kpt_000001.csc'),
fn_elsi_restart = None,
fn_geometry = Path(geo_unconstrained),
)

### Run 1D scan
run1d = False
x, y = 0, 10
if run1d:
    mode_indices= np.arange(45,48)
    ters.run_1d_multimode(
        working_dir = Path(f'ters1d_x{x:.1f}_y{y:.1f}'),
        mode_indices = mode_indices,
        tip_origin = (-0.000030, -1.696604, -4.6140),
        sys_origin = (0.0, 0.0, 0.0),
        tip_height = 4.0,
        xy_displacement = (x,y)
    )

### Choose points for 2D scan
def make_tippos(xs, ys):
    return list(zip(xs, ys))

def read_grid_coords(mode_idx: int, base_dir: Path = Path("ters2d")):
    """
    Read tip positions from all tippos_* folders of a given mode.
    Returns xs, ys as arrays.
    """
    mode_dir = base_dir / f"mode_{mode_idx:03d}"
    coords = []
    for tippos_dir in sorted(mode_dir.glob('tippos_*')):
        control_file = tippos_dir / 'positive_displacement' / 'field_on' / 'control.in'
        with open(control_file) as f:
            for line in f:
                if line.strip().startswith('rel_shift_from_tip'):
                    parts = line.split()
                    coords.append((float(parts[1]), float(parts[2])))
                    break
    coords = np.array(coords)
    return coords[:, 0], coords[:, 1]

# Option A: read grid from another mode's folder
#source_mode_idx = 0
#xs, ys = read_grid_coords(source_mode_idx)

# Option B: define manually
# -- Single point
xs, ys = [1], [0]

# -- Single line
#xs = np.linspace(-6, -1, 4)
#ys = np.zeros(4)

# -- Grid
# x = np.linspace(0, 15, 10)
# y = np.linspace(-1.5, -15, 9)
# X, Y = np.meshgrid(x, y)
# xs, ys = X.ravel(), Y.ravel()


### Run 2D scan
scan_range = (-15, 15, -15, 15)
run2d = True
if run2d:
    mode_indices = [9]
    for idx_mode in mode_indices:
        ters.run_2d_grid(
            working_dir = Path(f'ters2d'),
            idx_mode = idx_mode,
            tip_origin = (-0.000030, -1.696604, -4.6140),
            sys_origin = (0.0, 0.0, 0.0),
            tip_height = 4.0,
            tippos=make_tippos(xs, ys)
        )


### Copy constrained geometry.in into all calculation directories
main_dirs = []
if run1d:
    main_dirs.append(f"ters1d_x{x:.1f}_y{y:.1f}")
if run2d:
    main_dirs.append("ters2d")

paths = [f"{calc_dir}/mode_{idx_mode:03d}" for calc_dir in main_dirs for idx_mode in mode_indices]
MARKER = ".processed"

for path in paths:
    # Add constrained atoms back to the control.in and geometry.in
    for root, dirs, files in os.walk(path):
        if not dirs:
            marker_path = os.path.join(root, ".processed")
            if os.path.exists(marker_path):
                #print(f"Skipping already processed path: {root}")
                continue

            dest_geom = os.path.join(root, "geometry.in")
            with open(geo_constrained, "r") as f:
                geo_lines = f.readlines()
            with open(dest_geom, "r") as f:
                geom_lines = f.readlines()
            insert_index = 0
            for i, line in enumerate(geom_lines):
                if "lattice_vector" in line:
                    insert_index = i + 1  # insert after this line

            # Insert geo_lines at the desired position
            geom_lines = geom_lines[:insert_index] + geo_lines + geom_lines[insert_index:]
            # Write back to dest_geom
            with open(dest_geom, "w") as f:
                f.writelines(geom_lines)
            controlin = os.path.join(root, "control.in")
            with open(controlin, "a") as dest:
                for symbol in set(symbols_constrained) - set(symbols_unconstrained):
                    file_path = find_species_file(symbol, species_dir)
                    with open(file_path, "r") as src:
                        dest.writelines(src.readlines())

            with open(marker_path, "w") as f:
                f.write("processed\n")

    if run2d:
        # Make zerofield dirs
        pos_src = os.path.join(path, "tippos_000", "positive_displacement", "field_on")
        neg_src = os.path.join(path, "tippos_000", "negative_displacement", "field_on")
        pos_dst = os.path.join(path, "poszerofield")
        neg_dst = os.path.join(path,"negzerofield")

        if os.path.exists(pos_src) and not os.path.exists(pos_dst):
            shutil.copytree(pos_src, pos_dst, symlinks=True)
        if os.path.exists(neg_src) and not os.path.exists(neg_dst):
            shutil.copytree(neg_src, neg_dst, symlinks=True)

                
        pos_geom = os.path.join(pos_dst, "geometry.in")
        neg_geom = os.path.join(neg_dst, "geometry.in")
        zero_homogeneous_field(pos_geom)
        zero_homogeneous_field(neg_geom)
