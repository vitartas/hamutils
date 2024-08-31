import numpy as np

def get_rec_M(M, translations, k):
    rec_M = np.zeros((M.shape[1], M.shape[2]), dtype=np.complex128)
    phase_const = -2 * np.pi * 1j
    for idx, trans in enumerate(translations):
        phase = np.exp(phase_const * np.dot(k, trans))
        rec_M += M[idx] * phase
    return rec_M