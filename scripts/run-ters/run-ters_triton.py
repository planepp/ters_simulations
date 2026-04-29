from pathlib import Path
import numpy as np
import sys
import os
import shutil

sys.path.append(os.path.expanduser("~/.local/bin"))
from finite_field_ters import FiniteFieldTERS
from prepare_ters import (
    h, masses, geo_unconstrained, geo_constrained,
    species_dir, symbols_constrained, symbols_unconstrained,
    find_species_file
)

### Set up initial TERS object, fill with required data                     !!!
ters = FiniteFieldTERS(
hessian = h,
modes = None,
masses = masses,
dq = 5e-3,
efield = -1e-1,
storage_dir = Path('/home/planelp1/'),
fn_control_template = Path('ters_control_template.in'),
species_dir = Path(species_dir),
#fn_tip_groundstate = Path('zeros.cube'),
fn_tip_groundstate = None,
fn_tip_derivative = Path('tipA_05_vh_ft_0049_3221meV_x1000.cube'),
fn_elsi_restart = Path('D_spin_01_kpt_000001.csc'),
fn_geometry = Path(geo_unconstrained),
)

### Run multimode TERS calculation with the tip above the molecular COM    !!!
run1d = False
if run1d:
    ters.run_1d_multimode(
        mode_indices= np.arange(len(ters.modes)),
        tip_origin = (-0.000030, -1.696604, -4.6140),
        sys_origin = (0.0, 0.0, 0.0),
        tip_height = 4.0
    )

# test 2D infrastructure
run2d = True
if run2d:
    xx = 15
    yy = 15
    mode_indices = [151, 152, 153, 154]
    for idx_mode in mode_indices:
        ters.run_2d_grid(
            idx_mode= idx_mode,
            tip_origin = (-0.000030, -1.696604, -4.6140),
            sys_origin = (0.0, 0.0, 0.0),
            tip_height = 4.0,
            scan_range = (-xx, xx, -yy, yy),
            bins=(15,15)
        )

### Copy constrained geometry.in into all calculation directories
main_dirs = []
if run1d:
    main_dirs.append("ters1d")
if run2d:
    main_dirs.append("ters2d")

paths = [f"{calc_dir}/mode_{idx_mode:03d}" for calc_dir in main_dirs for idx_mode in mode_indices]

for path in paths:
    # Add constrained atoms back to the control.in and geometry.in
    for root, dirs, files in os.walk(path):
        if not dirs:
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

    # Make zerofield dirs
    pos_src = os.path.join(path, "tippos_000", "positive_displacement", "field_on")
    neg_src = os.path.join(path, "tippos_000", "negative_displacement", "field_on")
    pos_dst = os.path.join(path, "poszerofield")
    neg_dst = os.path.join(path,"negzerofield")

    if os.path.exists(pos_src):
        shutil.copytree(pos_src, pos_dst, dirs_exist_ok=True)
    if os.path.exists(neg_src):
        shutil.copytree(neg_src, neg_dst, dirs_exist_ok=True)    

    def zero_homogeneous_field(geom_path):
        with open(geom_path, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "homogeneous_field" in line:
                parts = line.split()
                lines[i] = f"{parts[0]} 0 0 0\n"
        with open(geom_path, "w") as f:
            f.writelines(lines)
            
    pos_geom = os.path.join(pos_dst, "geometry.in")
    neg_geom = os.path.join(neg_dst, "geometry.in")
    zero_homogeneous_field(pos_geom)
    zero_homogeneous_field(neg_geom)
