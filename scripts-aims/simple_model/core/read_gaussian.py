import numpy as np

def read_gaussian(file_name):
    """
    Read a Gaussian output file and return at least the vibrational frequencies,
    polarizabilities, and polarizability derivatives.

    Arguments:
        file_name: Path object -- the name of the Gaussian output file to read

    Returns:
        red_masses: np.ndarray -- the reduced masses
        frequencies: np.ndarray -- the vibrational frequencies of each mode
        polar_derivatives: np.ndarray -- the polarizability derivatives
        atom_an: np.ndarray -- ???
        atom_positions -- the cartesian coordinates of the atoms
        N1: int -- number of free atoms in a molecule
        N2: int -- number of fixed atoms in a molecule
    """

    def read_frequencies_red_masses(N1, N2, prefix):
        """
        Reads and returns vibrational frequencies and reduced masses from FHI-aims output.

        Arguments:
            N1: int -- number of free atoms
            N2: int -- number of fixed atoms
            prefix: str -- file prefix, e.g. 'fepc'

        Returns:
            frequencies: np.ndarray  shape (3*(N1-2),) or (3*N1-5,) if N1==2
            red_masses:  np.ndarray  same shape
        """
        import re

        # Determine number of modes (matching Gaussian convention: exclude 6 roto-translations)
        if N1 == 2 and N2 == 0:
            N_modes = 3 * N1 - 5
        else:
            N_modes = 3 * (N1 - 2)

        # --- Frequencies from <prefix>.Raman ---
        frequencies = []
        with open(f"fepc.Raman") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) == 4:
                    try:
                        frequencies.append(float(parts[1]))
                    except ValueError:
                        continue
        frequencies = np.array(frequencies)

        # --- Masses from masses.<prefix>.dat ---
        masses = []
        with open(f"masses.fepc.dat") as f:
            for line in f:
                parts = line.split()
                if parts:
                    try:
                        masses.append(float(parts[0]))
                    except ValueError:
                        continue
        masses = np.array(masses)          # shape (N1,)
        mass_vec = np.repeat(masses, 3)    # shape (3*N1,)

        # --- Eigenvectors from normalmodes.<prefix>.dat ---
        with open(f"normalmodes.fepc.dat") as f:
            content = f.read()

        blocks = re.split(r'Atoms:\s+\d+\s+Mode\s+#:\s*\d+[^\n]*\n', content)
        blocks = [b.strip() for b in blocks if b.strip()]

        red_masses = []
        for block in blocks:
            clean = block.replace('[', '').replace(']', '').replace(',', ' ')
            nums = []
            for token in clean.split():
                try:
                    nums.append(float(token))
                except ValueError:
                    pass
            if len(nums) >= 3 * N1:
                u = np.array(nums[:3 * N1])
                red_masses.append(1.0 / np.sum(u**2 / mass_vec))

        red_masses = np.array(red_masses)

        # Trim to N_modes (skip first 6 roto-translations, matching Gaussian)
        frequencies = frequencies[-N_modes:]
        red_masses  = red_masses[-N_modes:]

        return frequencies, red_masses 


    def read_atom_an(N1, N2, file):
        nat = N1 + N2
        n_dof = 3 * nat

        file = "eigen_vectors.fepc.dat"
        with open(file, "r") as f:
            data = np.fromstring(f.read(), sep=' ')

        n_modes = data.size // n_dof
        atom_an = data.reshape(n_modes, n_dof).T

        n_rigid = 5 if nat == 2 else 6
        return atom_an[:, n_rigid:]


    def read_polar_derivatives(N1, N2, file):
        """
        Computes polarizability derivatives from polarizability tensor derivatives
        and vibrational eigenvectors.

        Parameters
        ----------
        N1, N2 : int
            kept for compatibility (not used directly here)
        pol : np.ndarray
            Polarizability derivative tensor.
            Expected shape: (n_modes, 6) or (6, n_modes) depending on convention.
        eig_vec : np.ndarray
            Vibrational eigenvectors.
            Expected shape: (n_modes, n_modes) or (3N, n_modes)

        Returns
        -------
        dict with:
            alphasxx, alphasyy, alphaszz, alphasxy, alphasxz, alphasyz
            alpha : isotropic polarizability derivative
            beta  : Raman invariant
            alphas : full projected derivatives (6, n_modes)
        """

        beta = np.loadtxt("beta.dat")

        return beta

    
    def read_atom_positions(N1, N2, file):
        """
        Reads Cartesian coordinates of all atoms using ASE.
        """
        atoms = read("geometry.in", format="aims")

        return atoms.get_positions().ravel()


    def read_atomic_numbers(N1, N2, file):
        """
        Reads atomic numbers using ASE.
        """
        atoms = read("geometry.in", format="aims")
        return atoms.numbers.copy()


    from ase.io import read
    def read_molecule_constituent(file):
        """
        Returns:
            N1: number of free atoms
            N2: number of constrained atoms
        """
        file.seek(0)
        atoms = read("geometry.in", format="aims")

        constrained = set()

        if atoms.constraints:
            for c in atoms.constraints:
                if hasattr(c, "index"):
                    constrained.update(c.index)

        N2 = len(constrained)
        N1 = len(atoms) - N2

        return N1, N2


    # Reading the data from the Gaussian ouput file
    with open(file_name, 'r') as file:
        N1_N2 = read_molecule_constituent(file)
        if N1_N2 is None:
            return None
        N1, N2 = N1_N2

        frequencies_red_masses = read_frequencies_red_masses(N1, N2, file)
        atom_an = read_atom_an(N1, N2, file)
        polar_derivatives = read_polar_derivatives(N1, N2, file)
        atom_positions = read_atom_positions(N1, N2, file)
        atomic_numbers = read_atomic_numbers(N1, N2, file)
    
    parameters = [frequencies_red_masses, polar_derivatives, atom_an, atom_positions, atomic_numbers, N1, N2]
    if any(parameter is None for parameter in parameters):
        return None
    frequencies, red_masses = frequencies_red_masses
    unit_scale = 0.52917721092  # Converting units from bohr to A

    return red_masses, frequencies, polar_derivatives, atom_an, atom_positions, atomic_numbers, N1, N2


if __name__ == "__main__":
    result = read_gaussian("geometry.in")
