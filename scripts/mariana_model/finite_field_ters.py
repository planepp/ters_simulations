import os
from typing import Iterable, Union
from pathlib import Path
import pickle

import numpy as np
import scipy.constants as cs

import ase
import ase.io

class FiniteFieldTERS:
    """Object representation of a TERS calculation using finite homogeneous fields,
    including periodic boundary conditions.

    Here, the polarizability zz-component is accessed using finite differences from
    alpha_zz = d(mu) / dE ~ (mu(E) - mu(0)) / E.

    Given an optimized structure and its Hessian, the code sets up displacements,
    generates calculation directories, creates input files amd runs the single points.
    On output, the tabulated, position-dependent values of dipoles, polarizabilities and
    Raman intensities are dumped.
    """

    @property
    def frequencies(self):
        """Normal frequencies in atomic units"""
        return self._frequencies.copy()

    @property
    def wavenumbers(self):
        """Normal mode wavenumbers in common units (cm^-1)"""
        
        # convert a.u. to SI units (Hz)
        omega = self._frequencies.real #* cs.physical_constants['atomic unit of energy'][0] / cs.hbar

        # convert angular frequency to linear
        nu = omega / (2 * np.pi)

        # convert `nu` to wavenumbers in cm^-1
        wavenumbers = nu / cs.c
        wavenumbers *= 1e-2
        return wavenumbers

    @property
    def modes(self):
        """Normal mode vectors in atomic units"""
        return self._modes.copy()

    def __init__(
            self, 
            hessian: Union[None, np.ndarray],
            modes: Union[None, np.ndarray], 
            masses: np.ndarray, 
            dq: float,
            efield: float,
            storage_dir: Path,
            species_dir: Path,
            fn_control_template: Path,
            fn_tip_derivative: Path,
            fn_tip_groundstate: Path,
            fn_elsi_restart: Path,
            fn_geometry: str,
            cell: np.ndarray = None,
            submit_style: str = None,
            fn_batch: Path = None,
            ):
        """"""
        
        # parse ase geometry, define PBCs if requested
        system = ase.io.read(fn_geometry)
        if (system.pbc.all() == False) and (cell is not None):
            assert cell.shape == (3, 3)
            system.pbc = np.array([True] * 3)
            system.cell = cell
        self.system = system

        # check that we only have a hessian or modes, but not both, one of them must be `None`
        # this is to enable passing either a full hessian and taking care of the diag business here
        # or passing and using mode vectors directly as an input
        assert hessian is None or modes is None

        # unpack input
        self.hessian = hessian
        self.masses = masses
        self.dq = dq
        self.efield = efield
        self.parent_dir = Path.cwd()
        self.submit_style = submit_style
        self.fn_control_template = fn_control_template
        self.storage_dir = storage_dir
        self.species_dir = species_dir
        self.fn_batch = fn_batch
        self.fn_tip_derivative = fn_tip_derivative
        self.fn_tip_groundstate = fn_tip_groundstate
        self.fn_elsi_restart = fn_elsi_restart
        #self.fn_tip_no_field = fn_tip_no_field

        # assert that we only assign one float for the electric field, which will be in the z-direction
        assert isinstance(efield, float)

        if self.hessian is not None:
            # diagonalize hessian to get mode vectors
            # TODO: possibly symmetrize hessian before diagonalizing
            eigenvalues, modes = np.linalg.eigh(hessian)
            modes = modes.T
            frequencies = np.sqrt(eigenvalues.astype('complex'))
            self._frequencies = frequencies

        # lift mass-weighing from mode vectors, we need cartesian directions
        amu = cs.physical_constants['atomic mass constant'][0]
        mass_atomic = cs.physical_constants['atomic unit of mass'][0]
        amu_to_me = amu / mass_atomic
        masses = self.masses * amu_to_me
        # mass weighing is lifted and modes are normalized again
        # so that we can multiply by a given displacement in a chosen unit and have a displacement in the same unit
        # i.e., the normalized mode is just a cartesian direction
        modes /= np.sqrt(np.repeat(masses, 3))
        modes /= np.linalg.norm(modes, axis=1)[:, None]
        self._modes = modes

    def run_1d_multimode(
            self,
            working_dir: Path,
            mode_indices: Iterable, 
            tip_origin: Iterable, 
            sys_origin: Iterable, 
            tip_height: Iterable, 
            xy_displacement: tuple = (0,0), 
            dump_wavenumbers: bool = True
            ):
        """Wrapper around the `run` function to launch calculations with a single tip position for different modes"""
        
        # run over all modes
        existing_folders_count = 0
        for idx_mode in mode_indices:
            # make calculation directory
            calc_dir = self.parent_dir / working_dir / f'mode_{idx_mode:03d}'
            try:
                calc_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                existing_folders_count += 1
                #print(f"ERROR: Folder {calc_dir} already exists.")
            # run
            self._run(
                idx_mode=idx_mode,
                tip_origin=tip_origin,
                sys_origin=sys_origin,
                tip_height=tip_height,
                xy_displacement=xy_displacement,
                working_dir=calc_dir,
                ) 
        if existing_folders_count > 0: # short debug message
            print(f'ERROR: found {existing_folders_count} already existing folders')
        # optionally dump the wavenumbers into a pickle file
        if dump_wavenumbers:
            pickle.dump(self.wavenumbers, open('wavenumbers.pickle', 'wb'))        


    def run_2d_grid(
        self,
        working_dir: Path,
        idx_mode: int,
        tip_origin: Iterable,
        sys_origin: Iterable,
        tip_height: float,
        tippos: Iterable):
        
        """Run calculations for a list of (x, y) tip positions."""
        mode_dir = self.parent_dir / f'ters2d/mode_{idx_mode:03d}'
        existing = [int(d.name.split('_')[1]) for d in mode_dir.glob('tippos_*') if d.is_dir()]
        offset = max(existing) + 1 if existing else 0

        for i_calc, (x, y) in enumerate(tippos):
            calc_dir = mode_dir / f'tippos_{offset + i_calc:03d}'
            calc_dir.mkdir(parents=True, exist_ok=False)
            self._run(
                idx_mode=idx_mode,
                tip_origin=tip_origin,
                sys_origin=sys_origin,
                tip_height=tip_height,
                xy_displacement=(x, y),
                working_dir=calc_dir
                )


    def _run(
            self, 
            idx_mode: int,  
            tip_origin: Iterable,
            sys_origin: Iterable,
            tip_height: float,
            xy_displacement: Iterable,
            working_dir: Path
            ):
        """Run a single TERS calculation using FHI-aims, for a single mode and a single tip position.
        This entail perfoming 4 single point calculations for the two normal mode displacements and two E-field strengths.
        Arguments: 
            idx_mode: mode index according to the provided Hessian, must be chosen by user
            dq: Cartesian displacement, units are chosen by the user (modes are unitless), so this defines the unit fully.
            working_dir: working directory from which to build the calculation infrastructure and launch the calculation.
            add_zerofield: whether calculations with zero field strength should be performed. This is advisable with `run_1d_multimode`, but not with `run_2d_grid`, where only a single displacement is taken into account.
        """

        mode = self.modes[idx_mode]

        # prepare +- displaced geometries
        d = (self.dq * mode).reshape(-1, 3)
        pos_disp = self.system.positions + d
        neg_disp = self.system.positions - d

        # prepare all 4 directories and input files
        dir_pos = working_dir / 'positive_displacement'
        dir_neg = working_dir / 'negative_displacement'
        for d in (dir_pos, dir_neg):
            if self.fn_tip_groundstate is None:
                # only perform calculations with field on
                fieldtypes = ['field_on']
            else:
                # perform even zero-field calculations
                fieldtypes = ['field_on', 'zero_field']
            for fieldtype in fieldtypes:
                calc_dir = d / fieldtype
                calc_dir.mkdir(parents=True, exist_ok=True)
                # create control file with TERS parameters
                self._create_control(
                    calc_dir / 'control.in', 
                    self.storage_dir / self.fn_control_template,
                    self.species_dir,
                    self.fn_tip_derivative,
                    self.fn_tip_groundstate,
                    tip_origin, 
                    sys_origin,
                    tip_height, 
                    xy_displacement
                    )
                # create displaced geometry file with the correct homogeneous field setting
                if d == dir_pos:
                    disp_geometry = pos_disp
                elif d == dir_neg:
                    disp_geometry = neg_disp
                self._create_geometry(calc_dir / 'geometry.in', disp_geometry, fieldtype)
                # make a symlink to cube files so that it does not have to be copied, aims cannot do paths
                os.system(f"ln -sf {str(self.storage_dir / self.fn_tip_derivative):s} {str(calc_dir / self.fn_tip_derivative):s}")
                if self.fn_tip_groundstate is not None:
                    os.system(f"ln -sf {str(self.storage_dir / self.fn_tip_groundstate):s} {str(calc_dir / self.fn_tip_groundstate):s}")
                # make a symlink to ELSI restart file so that it does not have to be copied, aims cannot do paths
                if self.fn_elsi_restart is not None:
                    os.system(f"ln -sf {str(self.parent_dir / self.fn_elsi_restart):s} {str(calc_dir / self.fn_elsi_restart):s}")
                # copy batch script from parent directory and run
                if self.submit_style == 'slurm':
                    (calc_dir / 'run.sbatch').write_text(self.fn_batch.read_text()) # copy file without the need for shutil import
                    # this is a temporary measure to battle the cluster's job number limit - quite ugly
                    # TODO: think of somehthing better, avoiding platform specificity as much as possible
                    if os.path.isfile(calc_dir / 'aims.out'):
                        print(f"Skipping submit in {str(calc_dir):s}, FHI-aims output exists.")
                    else:
                        os.system(f"sbatch --job-name {str(calc_dir):s} --chdir {str(calc_dir):s} {str(calc_dir / 'run.sbatch'):s}")
                elif self.submit_style == 'draft':
                    (calc_dir / 'run.sbatch').write_text(self.fn_batch.read_text()) # copy file, but no shutil import
                #TODO: implement running aims outside of batch for personal PC use etc...
                elif self.submit_style == None: continue
                else:
                    raise NotImplementedError('Running outside of SLURM is not implemented at the moment.')


    def _create_control(
            self, 
            fn_target: Path, 
            fn_template: Path, 
            species_dir: Path,
            fn_cube_derivative: Path, 
            fn_cube_groundstate: Union[Path, None],
            tip_origin: Iterable, 
            sys_origin: Iterable, 
            tip_height: float, 
            xy_displacement: Iterable
            ):
        """Helper function to create the control.in file.
        Takes in a user-defined main section specifying the details of the electronic structure method
        and adds a block defining the TERS-related parameters.
        """
        # read in template 
        with open(fn_template, 'r') as f_in:
            fixedlines = f_in.readlines()

        # If system is non-periodic in all directions, remove k_grid lines
        if not self.system.pbc.all():
            fixedlines = [line for line in fixedlines if 'k_grid' not in line]

        # define list of new lines
        newlines = [
            f'pos_tip_origin          {tip_origin[0]:03f} {tip_origin[1]:03f} {tip_origin[2]:03f}\n',
            f'pos_sys_origin          {sys_origin[0]:03f} {sys_origin[1]:03f} {sys_origin[2]:03f}\n',
            f'tip_molecule_distance   {tip_height:03f}\n',
            f'rel_shift_from_tip      {xy_displacement[0]:03f} {xy_displacement[1]:0f}\n',
            f'nearfield_derivative    {str(fn_cube_derivative):s}\n'
        ] 
        
        # append groundstate tip filename if specified
        if fn_cube_groundstate is not None:
            newlines += f'nearfield_groundstate   {str(fn_cube_groundstate):s}\n'

        # get, read and paste basis functions
        uniquesymbols = []
        for symbol in self.system.get_chemical_symbols():
            if symbol not in uniquesymbols:
                uniquesymbols.append(symbol)

        basislines = []
        for symbol in uniquesymbols:
            fn_basis = list(species_dir.glob(f'*_{symbol:s}_default'))
            if len(fn_basis) == 0:
                raise ValueError('No consistent basis set files found.')
            elif len(fn_basis) > 1:
                raise ValueError('More than one basis set file names with consistent names found.')
            with open(fn_basis[0], 'r') as f_basis:
                basislines += f_basis.readlines()

        # concatenate lines and dump to target
        lines = fixedlines + newlines + basislines
        with open(fn_target, 'w') as f_out:
            f_out.writelines(lines)

 
    def _create_geometry(self, fn_target: Path, geometry: np.ndarray, fieldtype: str):
        """Helper function to create the displaced geometry.in file.
        Electric field is additionally specified.
        """

        # FIXME: could this be ASE-based?

        # geometry file has 4 components - cell, positions, E-field and z-level of vacuum. All have to be taken care of.
        # cell, only print lattice vectors if cell is given
        if self.system.pbc.all() == True:
            lines = [f'lattice_vector {a[0]:.16f} {a[1]:.16f} {a[2]:.16f}\n' for a in self.system.cell]
        else:
            lines = []
        # positions
        symbols = self.system.get_chemical_symbols()
        #lines += [f'atom {r[0]:.16f} {r[1]:.16f} {r[2]:.16f} {s}\n' for r, s in zip(geometry, symbols)]
        for r, s in zip(geometry, symbols):
            lines.append(f'atom {r[0]:.16f} {r[1]:.16f} {r[2]:.16f} {s}\n')
        # field
        if fieldtype == 'field_on':
            lines += [f'homogeneous_field 0.0 0.0 {self.efield:.16f}\n']
        elif fieldtype == 'zero_field':
            lines += [f'homogeneous_field 0.0 0.0 0.0\n']
        # vacuum level: we put this firmly into 49% of the height of the central cell, only set up if cell is given
        if self.system.pbc.all() == True:
            zlevel = 0.49 * self.system.cell[-1, -1]
            lines += [f'set_vacuum_level {zlevel:.16f}']

        # dump to target
        with open(fn_target, 'w') as f_out:
            f_out.writelines(lines)

def _read_aims_output(fn_aims: Path, periodic: bool):
    """Read the z-component of the dipole moment out of a single FHI-aims output file."""
    
    # initialize the dipole as nan and fill in a value later
    # this way the function always returns and does not explode if it finds nothing due to, e.g., an unconverged calculation 
    mu_z = np.nan
    
    # loop over the FHI-aims output
    with open(fn_aims) as f:
        lines = f.readlines()
        for line in lines:
            if periodic:
                if "| Total dipole moment in z-direction [eAng]" in line:
                    mu_z = float(line.split()[8])
            else:
                 if "| Total dipole moment [eAng]" in line:
                    mu_z = float(line.split()[8])
    return mu_z

def analyze_1d_ters(working_dir: Path, fn_wavenumbers: Path, efield: float, dq: float, periodic: bool):
    """Analysis function to gather data from a multimode TERS calculation."""

    fieldtypes = ['field_on', 'zero_field']
    displacementtypes = ['negative_displacement', 'positive_displacement']

    mode_dirs = sorted(working_dir.glob('mode_*'))
    mode_indices = [int(d.name.split('_')[1]) for d in mode_dirs if d.is_dir()]

    mode_indices_per_dtft = []
    fns_per_dtft = []
    dipoles = []
    for dt in displacementtypes:
        for ft in fieldtypes:
            fns = sorted(working_dir.glob(f'mode_*/{dt:s}/{ft:s}/aims.out'))
            mode_indices = []
            mu_z = []
            for fn in fns:
                idx = int(fn.parts[-4].split('_')[1])
                mu = _read_aims_output(fn, periodic=periodic)
                if mu is None or np.any(np.isnan(mu)):
            #        print(f"Warning: NaN or None in {fn}, excluding mode {idx}.")
                    continue
                mode_indices.append(idx)
                mu_z.append(mu)
            dipoles.append(mu_z)
            mode_indices_per_dtft.append(set(mode_indices))
            fns_per_dtft.append((fns, mode_indices))

    # Find mode indices present in ALL dt/ft combinations
    common_modes = mode_indices_per_dtft[0].intersection(*mode_indices_per_dtft[1:])
    skipped = mode_indices_per_dtft[0].union(*mode_indices_per_dtft[1:]) - common_modes
    if skipped:
        print(f"Warning: skipping mode indices {sorted(skipped)} (missing one displacement/field type).")

    common_modes_sorted = sorted(common_modes)
    filtered_dipoles = []
    for dtft_idx, (fns, mode_indices) in enumerate(fns_per_dtft):
        mask = [idx in common_modes for idx in mode_indices]
        filtered_dipoles.append([mu for mu, keep in zip(dipoles[dtft_idx], mask) if keep])

    dipoles = np.array(filtered_dipoles)
    # calculate polarizabilities
    alphas = np.array([(dipoles[i] - dipoles[i + 1]) for i in (0, 2)]) / efield
    # calculate d(alpha) / dQ
    dadq = (alphas[1] - alphas[0]) / (2 * dq)
    # calculate Raman intensities
    intensity = dadq**2
    mode_max = np.max(intensity)
    #intensity = intensity / mode_max
    #print(intensity)

    # read in wavenumbers
    wn = pickle.load(open(working_dir / fn_wavenumbers, 'rb'))

    return {
        'wavenumbers': wn[common_modes_sorted],
        'intensity': intensity,
        'd(alpha)/dQ': dadq,
        'alpha': alphas,
        'dipole': dipoles
    }


def analyze_2d_ters(working_dir: Path, mode_idx: list, efield: float, dq: float, periodic: bool, no_groundstate: bool = False, mu0_pos_displ: float = 0, mu0_neg_displ: float = 0):
    """Analysis function to gather data from a single mode, 2D TERS calculation."""
    
    '''
    if no_groundstate:
        try:
            mu0_pos_displ
        except NameError:
            print("If no GS Hartree cube was used, the value of the groundstate dipole must be supplied manually.")
        try:
            mu0_neg_displ
        except NameError:
            print("If no GS Hartree cube was used, the value of the groundstate dipole must be supplied manually.")
    '''
    displacementtypes = ['negative_displacement', 'positive_displacement']
    total_intensity = []
    intensity_max = 0
    if mode_idx is None:
        mode_patterns = ["mode*"]  # all
    elif isinstance(mode_idx, list):
        mode_patterns = [f"mode*{i:03d}" for i in mode_idx]
    else:
        raise ValueError("mode_idx must be None, an int, or a list of ints")
    mode_dirs = []
    for pattern in mode_patterns:
        mode_dirs.extend(sorted(working_dir.glob(pattern)))

    # remove duplicates (in case patterns overlap)
    for mode_dir in mode_dirs:
        dipoles = []
        dipoles_0 = []
        tippos_indices_per_dt = []
        fns_per_dt = []

        for dt in displacementtypes:
            # calculations with tip and field
            fns = sorted(mode_dir.glob(f'tippos_*/{dt:s}/field_on/aims.out'))
            tippos_indices = [int(fn.parts[-4].split('_')[1]) for fn in fns]
            mu_z = [_read_aims_output(fn, periodic=periodic) for fn in fns]
            dipoles.append(mu_z)
            tippos_indices_per_dt.append(set(tippos_indices))
            fns_per_dt.append((fns, tippos_indices))

            if not no_groundstate:
                # zero-field reference values
                fns_0 = sorted(mode_dir.glob(f'tippos_*/{dt:s}/zero_field/aims.out'))
                mu_z_0 = [_read_aims_output(fn_0, periodic=periodic) for fn_0 in fns_0]
                dipoles_0.append(mu_z_0)
            elif no_groundstate:
                if dt == displacementtypes[0]:
                    fn0_neg = mode_dir / 'negzerofield' / 'aims.out'
                    mu0_neg = _read_aims_output(fn0_neg, periodic=periodic)
                    dipoles_0.append([mu0_neg] * len(fns))
                elif dt == displacementtypes[1]:
                    fn0_pos = mode_dir / 'poszerofield' / 'aims.out'
                    mu0_pos = _read_aims_output(fn0_pos, periodic=periodic)
                    dipoles_0.append([mu0_pos] * len(fns))

        # Find tip positions present in ALL displacement types
        common_tippos = tippos_indices_per_dt[0] & tippos_indices_per_dt[1]
        skipped = tippos_indices_per_dt[0].symmetric_difference(tippos_indices_per_dt[-1])
        if skipped:
            print(f"Warning: skipping tip positions {sorted(skipped)} in "
                  f"{mode_dir.name} (missing one displacement type).")

        # Filter dipoles and dipoles_0 to only common tip positions
        filtered_dipoles = []
        filtered_dipoles_0 = []
        for dt_idx, (fns, tippos_indices) in enumerate(fns_per_dt):
            mask = [idx in common_tippos for idx in tippos_indices]
            filtered_dipoles.append([mu for mu, keep in zip(dipoles[dt_idx], mask) if keep])
            filtered_dipoles_0.append([mu for mu, keep in zip(dipoles_0[dt_idx], mask) if keep])

        dipoles = np.array(filtered_dipoles)
        dipoles_0 = np.array(filtered_dipoles_0)
        # calculate polarizabilities
        alphas = (dipoles - dipoles_0) / efield
        # calculate d(alpha)/dQ
        dadq = (alphas[1] - alphas[0]) / (2 * dq)
        # calculate Raman intensities
        mode_intensity = dadq**2
        #print(mode_intensity)
        #print(mode_intensity[0])
        mode_max = np.max(mode_intensity)
        #print(mode_max)
        #mode_intensity = mode_intensity / mode_max
        total_intensity.append(mode_intensity)

    return {
        'intensity': total_intensity,
        'tippos_indices': sorted(common_tippos), 
        'd(alpha)/dQ': dadq,
        'alpha': alphas,
        'dipole': dipoles,
        'dipole0': dipoles_0
    }
