import numpy as np

def get_energy_ordering(basis, energy_ordered_list):
    """
    Given a list of current energy-unordered basis, return an array of length(basis) containing
    the energy ordering of the current basis.
    """
    ordered_basis = np.array([convert_name_to_indices(name) for name in energy_ordered_list])

    ordering = np.zeros(basis.shape[0], dtype=np.int64)
    for i, bas in enumerate(basis):
        for j, ord_bas in enumerate(ordered_basis):
            if (bas[0], bas[1]) == (ord_bas[0], ord_bas[1]):
                ordering[i] = j
    return ordering

def turn_to_degenerate(mask, basis):
    """
    Expands the mask, e.g. [s, p] -> [s, px py pz]
    """
    l_array = basis[:,1]
    full_mask = [np.array([boolean for _ in range(2*l + 1)]) for (boolean, l) in zip(mask, l_array)]
    full_mask = np.concatenate(full_mask)
    return full_mask

def convert_name_to_indices(name):
    """
    e.g. 2p -> [2, 1], where n=1 (indexing in basis-indices starts as n_quantum), and l=1
    """
    letter_to_l_map = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
    n = int(name[0])
    l = letter_to_l_map[name[1]]
    return np.array([n, l])

def get_basis_core_mask(basis, lowest_valence_orb, energy_ordered_list, include_degen=True):
    """
    Get a list of booleans of length(basis), where the basis is not energy ordered
    """
    core_indices_list = []
    for orb in energy_ordered_list:
        if orb is lowest_valence_orb:
            break
        core_indices_list.append(convert_name_to_indices(orb))

    mask = np.array([any([(n, l) == (nc, lc) for nc, lc in core_indices_list]) for n, l in basis])
    if include_degen:
        mask = turn_to_degenerate(mask, basis)

    return mask

def get_basis_interaction_mask2d(basis_name1, basis_name2, basis):
    """
    e.g. for 4s and 5p get 4s5p mask
    """
    basis_f1 = convert_name_to_indices(basis_name1)
    basis_f2 = convert_name_to_indices(basis_name2)

    return _get_basis_interaction_mask2d(basis_f1, basis_f2, basis)

def _get_basis_interaction_mask2d(basis_f1, basis_f2, basis):

    n1, l1 = basis_f1
    n2, l2 = basis_f2

    # Construct non-degenerate masks
    mask1 = [(n1, l1) == (n, l) for n, l in basis]
    mask2 = [(n2, l2) == (n, l) for n, l in basis]

    # Construct degenerate masks
    mask1 = turn_to_degenerate(mask1, basis)
    mask2 = turn_to_degenerate(mask2, basis)

    return mask1, mask2
