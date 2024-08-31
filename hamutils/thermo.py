import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar

from ase.units import kB

def gaussian_smearing(E, mu, T):
    return 0.5 * (1 - erf((E - mu) / (kB * T)))

def fermi_dirac(E, mu, T):
    return 1 / (1 + np.exp((E - mu) / (kB * T)))

def fermi_dirac_integral(E, mu, T):
    return -1*kB*T*np.log(np.exp(-1 * (E - mu) / (kB * T)) + 1)

def get_number_density(E, DOS, mu, T, smearing):
    """
    Get number density for densities of states normalised per unit volume.
    """

    num_dens = 0
    for i in range(1, len(DOS)):
        integrand_i_minus_1 = DOS[i-1] * smearing(E[i-1], mu, T)
        integrand_i = DOS[i] * smearing(E[i], mu, T)

        # Trapezoid
        num_dens += 0.5*(integrand_i + integrand_i_minus_1) * (E[i] - E[i-1])

    return num_dens

def _chemical_potential_loss_function(target_dens, E, DOS, T, smearing):
    """
    Construct loss function used in optimising chemical potential.
    """
    def loss_function(mu):
        return np.abs(target_dens - get_number_density(E, DOS, mu, T, smearing))
    return loss_function

def get_chemical_potential(target_dens, E, DOS, T, smearing=fermi_dirac):
    """
    Get chemical potential by optimisation based on the target electron density.
    """
    lf = _chemical_potential_loss_function(target_dens, E, DOS, T, smearing)
    res = minimize_scalar(lf)
    return res.x

def get_grand_potential_density(E, DOS, mu, T):
    """
    Obtain band Grand potential density per unit volume (assuming DOS is normalised per volume).
    """
    grand_pot = 0

    for i in range(1, len(DOS)):
        integrand_i_minus_1 = DOS[i-1] * fermi_dirac_integral(E[i-1], mu, T)
        integrand_i = DOS[i] * fermi_dirac_integral(E[i], mu, T)

        # Trapezoid
        grand_pot += 0.5*(integrand_i + integrand_i_minus_1) * (E[i] - E[i-1])

    return grand_pot

def get_entropy_density(E, DOS, mu, T):
    """
    Get band entropy density per unit volume (assuming DOS is normalised per volume).
    """
    entropy = 0

    for i in range(1, len(DOS)):
        integrand_i_minus_1 = DOS[i-1] * ((E[i-1] - mu)*fermi_dirac(E[i-1], mu, T) - fermi_dirac_integral(E[i-1], mu, T))
        integrand_i = DOS[i] * ((E[i] - mu)*fermi_dirac(E[i], mu, T) - fermi_dirac_integral(E[i], mu, T))

        # Trapezoid
        entropy += 0.5*(integrand_i + integrand_i_minus_1) * (E[i] - E[i-1])

    return entropy

def get_helmholtz_free_energy_density(mu, phi, num_dens):
    """
    Get band Helmholtz free energy density from estimated chemical potential and Grand potential density.
    """
    return phi + num_dens * mu