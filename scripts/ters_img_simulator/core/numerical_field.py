import numpy as np
def numerical_field(atom_polarizabilities, atoms, frequencies, E_loc):
    """
    Calculates and returns intensities of each vibrational mode and polarized dipoles of each 
    atom for each vibrational mode, given an externally provided local field.

    Arguments:
    atom_polarizabilities: np.ndarray -- the polarizability of each atom
    atoms: int -- the number of atoms in the molecule
    frequencies: np.ndarray -- the vibrational frequencies of each mode (unit: cm^-1)
    E_loc: np.ndarray -- local field at each atomic position, shape (atoms, 3) (unit: A)
 
    Returns:
        intensity: np.ndarray -- intensity of each vibrational mode
        dipole: np.ndarray -- polarized dipole of each atom for each mode
    """
    E_local = E_loc.T.reshape(3*atoms)

    # Calculating mode intensities and dipoles
    mode_intensities = np.zeros((3*atoms-6, 2))
    dipoles = np.tensordot(atom_polarizabilities, E_local, axes=([2], [0]))
    E_scattering = np.einsum('i,ij->j', E_local, dipoles.T)

    mode_intensities[:, 0] = frequencies
    mode_intensities[:, 1] = E_scattering**2

    return mode_intensities, dipoles
