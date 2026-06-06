import pickle
from pathlib import Path

import numpy as np

from finite_field_ters import FiniteFieldTERS

# masses (TCNE)
# masses = np.array([12.011] * 32 + [14.007] * 8 + [1.008] * 16 + [65.38])

mass_dict = {
    "H": 1.00794,
    "C": 12.0107,
    "N": 14.0067,
    "Zn": 65.409,
}

masses = []

with open("geometry.in", "r") as f:
    for line in f:
        if line.strip().startswith("atom"):
            symbol = line.split()[4]
            masses.append(mass_dict[symbol])

masses = np.array(masses)

# read in hessian
#h = pickle.load(open('hessian.pickle', 'rb'))
h = np.loadtxt("hessian.atif100.dat")

# set up initial TERS object, fill with required data
ters = FiniteFieldTERS(
hessian = h,
modes = None,
masses = masses,
dq = 5e-3,
efield = -1e-1,
#submit_style = 'slurm',
submit_style = 'draft',
fn_control_template = Path('template.in'),
aims_dir = Path('/scratch/project_2001912/species_defaults/light/'),
fn_batch = Path('script.sh'),
fn_tip_groundstate = Path('zeros.cube'),
fn_tip_derivative = Path('tipA_05_vh_ft_0049_3221meV_x1000.cube'),
fn_elsi_restart = None,
fn_geometry = Path('geometry.in'),
)

# run multimode TERS calculation with the tip above the molecular COM
ters.run_1d_multimode(
mode_indices= np.arange(34, 154),    #np.arange(len(ters.modes)),
tip_origin = (-0.000030, -1.696604, -4.6140),
sys_origin = (0.0, 0.0, 0.0),
tip_height = 4.0,
xy_displacement = (0.0, 0.0),
dump_wavenumbers=True
)
