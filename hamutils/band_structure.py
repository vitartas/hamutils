from hamutils.misc import vector_str
from hamutils.fourier import get_rec_M
from hamutils.const import HARTREE_TO_EV

import os
import glob

import numpy as np
import scipy.linalg

from ase.dft.kpoints import bandpath


def get_band_input(cell, density):
    """
    Get band input for FHI-aims using Setyawan-Curtarolo convention

    density: float
      number of points per reciprocal space
    """
    lattice = cell.get_bravais_lattice()
    band_input_list = []
    special_paths = lattice.special_path.split(",")
    for special_path in special_paths:
        for i, point_end in enumerate(special_path[1:]):
            point_start = special_path[i]
            k_start = lattice.get_special_points()[point_start]
            k_end = lattice.get_special_points()[point_end]

            n_points = len(bandpath(f"{point_start}{point_end}", cell, density=density).kpts)

            band_input = f"band {vector_str(k_start)} {vector_str(k_end)} {n_points} {point_start} {point_end}"
            band_input_list.append(band_input)

    return band_input_list

def compute_band(H, S, translations_H, translations_S, karray, diag_mode):
    """
    karray: Nk x 3
    H: C x M x M

    band_data: Nk x M
    """
    band_data = np.zeros((karray.shape[0], H.shape[-1]))
    for i, kpoint in enumerate(karray):
        rec_H = get_rec_M(H, translations_H, kpoint)
        rec_S = get_rec_M(S, translations_S, kpoint)

        if diag_mode == "general":
            eigenvals = scipy.linalg.eigvals(rec_H, rec_S)
            band_data[i, :] = np.sort(np.real(eigenvals))
        elif diag_mode == "hermitian":
            eigenvals = scipy.linalg.eigvalsh(rec_H, rec_S)
            band_data[i, :] = eigenvals
        else:
            assert "Invalid diagonalisation method"

    return band_data

def write_band_structure(H, S, translations_H, translations_S, band_input, direc, diag_mode="general"):
    """
    band_input: FHI-aims band input
    direc: where to write the data
    """
    if not os.path.exists(direc):
        os.mkdir(direc)

    for i, line in enumerate(band_input):
        line = line.split()
        k_start = np.array([np.float64(line[i]) for i in range(1, 4)])
        k_end = np.array([np.float64(line[i]) for i in range(4, 7)])
        nk = np.int64(line[7])

        karray = np.zeros((nk, 3))
        kstep = (k_end - k_start) / (nk - 1)
        for ik in range(nk):
            karray[ik, :] = k_start + ik * kstep
        
        band_data = compute_band(H, S, translations_H, translations_S, karray, diag_mode)

        with open(os.path.join(direc, f"cband{i}.dat"), "w") as f:
            for ik, line in enumerate(band_data):
                f.write(f"{' '.join(map(str, karray[ik]))} {' '.join(map(str, line))}\n")

    return None

def read_bands(path, fermi_level, mode):
    """
    Get eigenvalues of bands along high-symmetry paths.
    All bands are returned which are ordered in the same way as lattice.special_path
    (unless surface bands were computed, in which case not all possible bands will be in the files)
    """
    if mode == "custom":
        return read_bands_custom(path, fermi_level)
    elif mode == "ACEh":
        return read_bands_aceh(path, fermi_level)
    elif mode == "FHI-aims":
        return read_bands_fhiaims(path)

def read_bands_custom(path, fermi_level):
    """
    Bands from custom real-space -> band structure implementation.
    """
    n_irrel_columns = 3
    band_paths = sorted(glob.glob(os.path.join(path, "cband*")))
    n_bands = np.loadtxt(band_paths[0]).shape[1] - n_irrel_columns

    data_tot = []
    for i, band_path in enumerate(band_paths):
        data = np.loadtxt(band_path)
        rel_data = np.zeros((data.shape[0], n_bands))
        for i in range(n_bands):
            # Assuming we can use the same Ha->eV conversion as for FHI-aims
            # (as we assume the data was generated using FHI-aims)
            rel_data[:,i] = data[:,n_irrel_columns + i] * HARTREE_TO_EV - fermi_level
        data_tot.append(rel_data)

    return data_tot

def read_bands_aceh(path, fermi_level):
    """
    The ACEh-computed bands are in Hartree and are not shifted with respect to the Fermi level.
    This function reads the files and scales+shifts the band structure accordingly.

    fermi_level: Chemical potential from FHI-aims calculation in eV
    """
    # Need different sorting for cases like [band_1, band_10, band_2]
    band_paths = sorted(glob.glob(os.path.join(path, "band*")), key=get_band_idx)
    n_irrel_columns = 3
    n_bands = np.loadtxt(band_paths[0]).shape[1] - n_irrel_columns

    data_tot = []
    for i, band_path in enumerate(band_paths):
        data = np.loadtxt(band_path)
        rel_data = np.zeros((data.shape[0], n_bands))
        for i in range(n_bands):
            rel_data[:,i] = data[:,n_irrel_columns + i] * HARTREE_TO_EV - fermi_level
        data_tot.append(rel_data)

    return data_tot

def read_bands_fhiaims(path):
    """
    Bands from FHI-aims output bands method. Note that the bands are already in eV and shifted wrt eF
    """
    n_irrel_columns = 4
    band_paths = sorted(glob.glob(os.path.join(path, "band1*")))
    n_bands = int((np.loadtxt(band_paths[0]).shape[1] - n_irrel_columns) / 2)

    data_tot = []
    for i, band_path in enumerate(band_paths):
        data = np.loadtxt(band_path)
        rel_data = np.zeros((data.shape[0], n_bands))
        for i in range(n_bands):
            rel_data[:,i] = data[:,n_irrel_columns + 2*i + 1]
        data_tot.append(rel_data)

    return data_tot

def get_band_idx(filename):
    return int(filename.split("_")[-1].split(".")[0])

def get_band_xticks(cell):
    lattice = cell.get_bravais_lattice()
    special_paths = lattice.special_path.split(",")

    xtick_str = ""
    for special_path in special_paths:
        for char in special_path:
            xtick_str += f"{char} "
        xtick_str = xtick_str[:-1]

    return xtick_str.split()