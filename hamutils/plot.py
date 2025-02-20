from hamutils.mask import _get_basis_interaction_mask2d
from hamutils.band_structure import get_band_xticks

import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
from typing import Optional

from ase.cell import Cell

diverging_colors = np.tile(["#443F90", "#685BA7", "#A599CA", "#F5DDEB", "#F592A5", "#EA6E8A", "#D21C5E"], 10)
qualitative_colors = np.tile(["#FF1F5B", "#00CD6C", "#009ADE", "#AF58BA", "#FFC61E", "#F28522"], 10)
matplotlib_colors = np.tile([u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
                             u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'], 10)

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=qualitative_colors)
mpl.rcParams['figure.dpi'] = 400
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['axes.linewidth'] = 1.2

def find_island_corners(array):
    corners_list = []
    visited = np.zeros_like(array, dtype=bool)
    nrows, ncols = array.shape
    
    for i in range(nrows):
        for j in range(ncols):
            if array[i, j] == 1 and not visited[i, j]:
                # Find the bounds of the current island
                x1, y1 = i, j
                x2, y2 = i, j
                
                while x2 + 1 < nrows and array[x2 + 1, j] == 1:
                    x2 += 1
                while y2 + 1 < ncols and array[i, y2 + 1] == 1:
                    y2 += 1
                
                # Mark the island as visited
                visited[x1:x2 + 1, y1:y2 + 1] = True
                
                # Get the corners of the island
                corners = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
                corners_list.append(corners)
    
    return corners_list

def plot_frame_around(ax, mask_matrix):
    corners_list = find_island_corners(mask_matrix)
    for corners in corners_list:
        w = 0.5
        row_path = [row for row, _ in corners]
        row_path = [row_path[0]-w, row_path[1]-w, row_path[2]+w, row_path[3]+w]
        col_path = [col for _, col in corners]
        col_path = [col_path[0]-w, col_path[1]+w, col_path[2]+w, col_path[3]-w]
        ax.fill(row_path, col_path, edgecolor="black", fill=False)

def plot_frame_around_subblocks(ax, basis):
    for basis_fi in basis:
        for basis_fj in basis:
            mask_fi, mask_fj = _get_basis_interaction_mask2d(basis_fi, basis_fj, basis)
            mask2d = np.tensordot(mask_fi, mask_fj, axes=0)    
            plot_frame_around(ax, mask2d)
    return None

def plot_band_structure(
        ax: Axes,
        band_plot_dict: dict[str, dict],
        cell: Optional[Cell] = None,
        add_ticks: bool = True
    ) -> None:
    """
    Plot band structure in axis `ax`.

    ## Arguments ##
        `ax`: Matplotlib axis object.
        `band_plot_dict`: Dictionary of dictionaries containing information about the band.
        `cell`: ASE cell object of the plotted configuration, only required if `add_ticks` is True.
        `add_ticks`: if True, all the high symmetry points of the brillouin lattice will be used as ticks.
    
    ## Example ##
    ```
    band_plot_dict = {'True': {'data': band_structure_data, 'style': {'color': 'black'}}}
    ```
    """
    for band_key in band_plot_dict.keys():

        bandpath_data_list = band_plot_dict[band_key]["data"]

        x_counter_list = [0]
        x_counter = 0
        for i_bandpath, bandpath_data in enumerate(bandpath_data_list):
            Nk_bandpath = bandpath_data.shape[0]
            Nbands = bandpath_data.shape[1]

            for i_band in range(Nbands):
                first_band = (i_bandpath == 0) and (i_band == 0)

                ax.plot(range(x_counter, x_counter + Nk_bandpath), bandpath_data[:, i_band],
                        **band_plot_dict[band_key]["style"],
                        label=f"{band_key}" if first_band else None)

                if "std" in band_plot_dict[band_key]:
                    std_dict = band_plot_dict[band_key]["std"]
                    bandpath_std = std_dict["data"][i_bandpath]
                    label = band_plot_dict[band_key]["std"].get("label", f"{band_key} Uncertainty")
                    ax.fill_between(range(x_counter, x_counter + Nk_bandpath),
                                    bandpath_data[:, i_band] - 2 * bandpath_std[:, i_band],
                                    bandpath_data[:, i_band] + 2 * bandpath_std[:, i_band],
                                    **std_dict["style"],
                                    label=label if first_band else None)

            # minus one because the adjacent special k-point path ends overlap
            x_counter += Nk_bandpath - 1
            x_counter_list.append(x_counter)

    for x in x_counter_list[1:-1]:
        ax.axvline(x, color="black")

    ax.set_xlim(x_counter_list[0], x_counter_list[-1])

    if add_ticks:
        ax.set_xticks(np.array(x_counter_list), get_band_xticks(cell))

    ax.set_ylabel(r"$E - \epsilon_{F}$ / eV")

    return x_counter_list
