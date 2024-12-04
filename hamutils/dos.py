from hamutils.kgrid import get_kgrid
from hamutils.fourier import get_rec_M_batch
from hamutils.const import HARTREE_TO_EV
from time import time
from multiprocessing import Pool

import os
import numpy as np
import scipy.linalg

def gaussian_broadening(E, eigenv, sigma):
    return np.exp(-((E - eigenv) / sigma)**2) / (np.sqrt(np.pi) * sigma)

def compute_dos(cell, energy_range, n_points, fermi_level, H, S, translations, kpoint_density,
                broadening=0.1, convert_to_eV=True):
    """
    Compute DOS using real-space matrices
    """
    if convert_to_eV:
        conversion = HARTREE_TO_EV
    else:
        conversion = 1.0

    kgrid = get_kgrid(cell=cell, k_grid_density=kpoint_density)

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

    for i, E in enumerate(dos_energies):
        dos_values[i] = np.sum(gaussian_broadening(E, all_eigenvals_eV_fermi, broadening))

    # BvK
    dos_values /= len(kgrid)
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