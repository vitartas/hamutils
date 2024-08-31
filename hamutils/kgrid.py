import numpy as np

def get_kgrid(cell, k_grid_density, type="gamma", flatten=True):
    """
    Return a grid of shape (3 x Nb1 x Nb2 x Nb3) containing "normalised" k-points.
    The resulting grid is Gamma-centered.

    If flatten, reshape the grid to a shape (nk x 3)
    """
    
    # Need to multiply by 2pi as ASE does not include it
    rec_cell = 2 * np.pi * cell.reciprocal()
    b_lengths = [np.linalg.norm(b) for b in rec_cell]
    n_k_grid = [np.int64(np.ceil(b_length * k_grid_density)) for b_length in b_lengths]

    # k-point spacings
    dks = [1 / nk for nk in n_k_grid]

    # Need to construct a "normalised" k-point grid that would span each b vector from 0 to 1.
    if type == "gamma":
        kgrid = np.meshgrid(*[np.linspace(0, 1, nk+1)[:-1] for nk in n_k_grid])
        kgrid = np.array(kgrid)

    if flatten:
        m, n, o, p = kgrid.shape
        nk = n * o * p
        kgrid = kgrid.reshape(m, nk).T

    return kgrid

def get_nkstr_from_density(cell, k_grid_density):
    """
    Compute nkx nky nkz string from supplied k_grid_density.
    """

    # Need to multiply by 2pi as ASE does not include it
    rec_cell = 2 * np.pi * cell.reciprocal()

    b_lengths = [np.linalg.norm(b) for b in rec_cell]
    nk_list = [np.int64(np.ceil(b_length * k_grid_density)) for b_length in b_lengths]

    return f"{nk_list[0]} {nk_list[1]} {nk_list[2]}"