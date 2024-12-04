import numpy as np
from numba import njit, prange

def get_rec_M(M, translations, k):
    # \exp{ -2\pi i \hat{k}T^{T} }
    phase_const = -2 * np.pi * 1j
    phase_array = np.exp(phase_const * np.einsum("i,ti->t", k, translations))

    rec_M = np.einsum("t,tij->ij", phase_array, M)
    return rec_M

def get_rec_M_batch(M, translations, k_array, numba=True):
    if numba:
        # Convert k_array into contiguous `C` memory (row-major)
        k_array = np.array(k_array)
        translations = np.asarray(translations, dtype=np.float64)
        return _get_rec_M_batch_numba(M, translations, k_array)
    else:
        return _get_rec_M_batch(M, translations, k_array)

def _get_rec_M_batch(M, translations, k_array):
    """
    Computes a batch of reciprocal M matrices from
    translations Nt x 3 and k_array Nk x 3
    """
    # \exp{ -2\pi i \hat{K}T^{T} }
    phase_const = -2 * np.pi * 1j
    phase_array = np.exp(phase_const * np.einsum("ki,ti->kt", k_array, translations))

    rec_M_array = np.einsum("kt,tij->kij", phase_array, M)
    return rec_M_array

@njit(parallel=True)
def _get_rec_M_batch_numba(M, translations, k_array):
    Nt, Nb, Nb = M.shape
    Nk, _ = k_array.shape

    phase_const = -2 * np.pi * 1j
    phase_array = k_array @ translations.T

    phase_array_complex = np.asarray(phase_array, np.complex128)
    phase_array_complex = np.exp(phase_const * phase_array_complex)

    rec_M_array = np.zeros((Nk, Nb, Nb), dtype=np.complex128)
    for ik in range(Nk):
        for it in range(Nt):
            rec_M_array[ik, :, :] += phase_array_complex[ik, it] * M[it, :, :]

    return rec_M_array