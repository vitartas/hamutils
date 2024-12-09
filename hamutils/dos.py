from hamutils.kgrid import get_kgrid
from hamutils.fourier import get_rec_M_batch
from hamutils.const import HARTREE_TO_EV

import os
import numpy as np
import scipy.linalg

def gaussian_broadening(dE, sigma):
    return np.exp(-(dE / sigma)**2) / (np.sqrt(np.pi) * sigma)

def compute_dos(cell, energy_range, n_points, fermi_level, H, S, translations, kpoint_density,
                broadening=0.1, convert_to_eV=True, n_batches=10):
    """
    Compute DOS using real-space matrices
    """
    if convert_to_eV:
        conversion = HARTREE_TO_EV
    else:
        conversion = 1.0

    kgrid = get_kgrid(cell=cell, k_grid_density=kpoint_density)
    Nk = kgrid.shape[0]

    rec_H_batch = get_rec_M_batch(H, translations, kgrid)
    rec_S_batch = get_rec_M_batch(S, translations, kgrid)

    all_eigenvals_map = map(scipy.linalg.eigvalsh, rec_H_batch, rec_S_batch)
    all_eigenvals = np.concatenate(list(all_eigenvals_map))
    # Nk = kgrid.shape[0]
    # chunksize = Nk // n_proc
    # with Pool(processes=n_proc) as pool:
    #     all_eigenvals = pool.starmap(scipy.linalg.eigvalsh, zip(rec_H_batch, rec_S_batch), chunksize=chunksize)
    # all_eigenvals = np.concatenate(all_eigenvals)

    all_eigenvals_eV_fermi = all_eigenvals * conversion - fermi_level

    dos_energies = np.linspace(energy_range[0], energy_range[-1], n_points)
    dos_values = np.zeros_like(dos_energies)

    for i in range(n_points):
        dE = dos_energies[i] - all_eigenvals_eV_fermi
        dos_values[i] = np.sum(gaussian_broadening(dE, broadening))

    # BvK
    dos_values /= Nk
    # 2 els per orbital
    dos_values *= 2

    return dos_energies, dos_values

def compute_dos_from_eigenvals(all_eigenvals, energy_range, n_points, fermi_level, Nk,
                               broadening=0.1, convert_to_eV=True):
    """
    Method to compute DOS from supplied eigenvalues, does not require to perform diagonalisation.
    """
    if convert_to_eV:
        conversion = HARTREE_TO_EV
    else:
        conversion = 1.0

    all_eigenvals_eV_fermi = all_eigenvals * conversion - fermi_level
    
    dos_energies = np.linspace(energy_range[0], energy_range[-1], n_points)
    dos_values = np.zeros_like(dos_energies)

    for i in range(n_points):
        dE = dos_energies[i] - all_eigenvals_eV_fermi
        dos_values[i] = np.sum(gaussian_broadening(dE, broadening))

    # BvK
    dos_values /= Nk
    # 2 els per orbital
    dos_values *= 2

    return dos_energies, dos_values

def write_dos(cell, energy_range, n_points, fermi_level, H, S, translations, kpoint_density,
              direc, broadening=0.1, convert_to_eV=True):
    """
    Compute and write DOS to a file "cdos.dat" in the directory direc.
    Two columns: 1st energy, 2nd DOS value 
    """
    
    if not os.path.exists(direc):
        os.makedirs(direc, exist_ok=True)

    dos_energies, dos_values = compute_dos(
        cell, energy_range, n_points, fermi_level, H, S, translations, kpoint_density, 
        broadening, convert_to_eV
    )

    np.savetxt(os.path.join(direc, "cdos.dat"), np.stack([dos_energies, dos_values]).T)

    return None