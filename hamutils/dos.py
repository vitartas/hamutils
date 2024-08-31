from hamutils.kgrid import get_kgrid
from hamutils.fourier import get_rec_M
from hamutils.const import HARTREE_TO_EV

import os
import numpy as np
import scipy.linalg

def gaussian_broadening(E, eigenv, sigma):
    return np.exp(-((E - eigenv) / sigma)**2) / (np.sqrt(np.pi) * sigma)

def compute_dos(cell, energy_range, n_points, fermi_level, H, S, translations, kpoint_density, broadening=0.1):
    """
    Compute DOS using real-space matrices
    """
    kgrid = get_kgrid(cell, kpoint_density)
    nk = len(kgrid)
    Nb = H.shape[-1]

    all_eigenvals = np.zeros((nk * Nb,))
    for ik, kpoint in enumerate(kgrid):
        rec_H = get_rec_M(H, translations, kpoint)
        rec_S = get_rec_M(S, translations, kpoint)

        eigenvals = scipy.linalg.eigvalsh(rec_H, rec_S)
        all_eigenvals[ik * Nb : (ik + 1) * Nb] = eigenvals

    all_eigenvals_eV_fermi = all_eigenvals * HARTREE_TO_EV - fermi_level

    dos_energies = np.linspace(energy_range[0], energy_range[-1], n_points)
    dos_values = np.zeros_like(dos_energies)

    for i, E in enumerate(dos_energies):
        dos_values[i] += np.sum(gaussian_broadening(E, all_eigenvals_eV_fermi, broadening))

    # BvK
    dos_values /= len(kgrid)
    # 2 els per orbital
    dos_values *= 2

    return dos_energies, dos_values    

def write_dos(cell, energy_range, n_points, fermi_level, H, S, translations, kpoint_density,
              direc, broadening=0.1):
    """
    Compute and write DOS to a file "cdos.dat" in the directory direc.
    Two columns: 1st energy, 2nd DOS value 
    """
    
    if not os.path.exists(direc):
        os.mkdir(direc)

    dos_energies, dos_values = compute_dos(
        cell, energy_range, n_points, fermi_level, H, S, translations, kpoint_density, broadening
    )

    np.savetxt(os.path.join(direc, "cdos.dat"), np.stack([dos_energies, dos_values]).T)

    return None