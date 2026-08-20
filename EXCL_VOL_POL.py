### Method 1 with excluded volume contribution
### Macromolecules 29, 23, 7602-7612, 1996

import numpy as np
import jscatter as js
from scipy.special import jv, sici, gamma
from scipy.integrate import simpson
from chi_coefs import *

def Gauss(R, R0=50, sigma=5):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((R - R0) / sigma)**2)

def Si(x):
    return sici(x)[0]

def S_rod(q, L):
    qL = q * L
    # Suppress division by zero warnings and fix NaN singularity at q=0
    return np.where(qL == 0, 1.0, 2 * Si(qL) / qL - 4 * (np.sin(qL / 2))**2 / (qL**2))

def gauss_chain(q, R_g):
    x = (R_g * q)**2
    return np.where(x == 0, 1.0, 2 * (np.exp(-x) - 1 + x) / (x**2))

def cylinder(q, R0, sigmaR):
    # Reduced integration grid points from 500 to 80 for vast speed improvements
    R = np.linspace(max(1e-2 * R0, R0 - 5 * sigmaR), R0 + 5 * sigmaR, 80)
    rad_dist = Gauss(R, R0, sigmaR)
    qR = q[:, np.newaxis] * R
    with np.errstate(divide='ignore', invalid='ignore'):
        form_factor = (2 * jv(1, qR) / qR)**2
        form_factor = np.where(qR == 0, 1.0, form_factor)

    Int = np.sum(form_factor * rad_dist, axis=1)
    return Int / np.max(Int)

def get_Rg_excl_vol(L, b):
    n_b = L / b
    R_g0_2 = ((L * b) / 6)
    alpha_2 = (1 + (n_b / 3.12)**2 + (n_b / 8.67)**3)**(0.170 / 3)
    return np.sqrt(R_g0_2 * alpha_2)

def xi_func(q, L, b):
    R_g = get_Rg_excl_vol(L, b)
    return q * b * (np.pi * b / (1.103 * L))**(1.5) * (R_g / b)**(2 * 1.282)

def chi_func(q, L, b):
    xi = xi_func(q, L, b)
    return np.exp(-xi**(-5))

def Gamma_func(q, L, b):
    chi = chi_func(q, L, b)
    xi = xi_func(q, L, b)
    
    # Vectorized loop equivalent
    sum1 = (get_Ai(2, L, b) * xi**2 + 
            get_Ai(3, L, b) * xi**3 + 
            get_Ai(4, L, b) * xi**4 + 
            get_Ai(5, L, b) * xi**5)
            
    sum2 = (get_Bi(0, L, b) + 
            get_Bi(1, L, b) * xi**-1 + 
            get_Bi(2, L, b) * xi**-2)

    return 1 + (1 - chi) * sum1 + chi * sum2

def excl_vol(q, L, b):
    R_g = get_Rg_excl_vol(L, b)
    C1 = 1.220; C2 = 0.4288; C3 = -1.651; C4 = 1.523; C5 = 0.1477
    nu = 0.588

    qRg = q * R_g
    w = (1 + np.tanh((qRg - C4) / C5)) / 2
    sum_terms = C1 * qRg**(-1 / nu) + C2 * qRg**(-2 / nu) + C3 * qRg**(-3 / nu)

    return (1 - w) * gauss_chain(q, R_g) + w * sum_terms

def S_full(q, L, b):
    S_c = excl_vol(q, L, b)
    chi = chi_func(q, L, b)
    gamma = Gamma_func(q, L, b)
    rod = S_rod(q, L)
    return ((1 - chi) * S_c + chi * rod) * gamma

def polymer_chains_excl_vol(q, L, b):
    return S_full(q, L, b)

def Shulz_Zimm(N0, sigma):
    _a_min = -np.log10(N0)
    # Reduced integration grid from 400 to 100 
    a = np.logspace(_a_min, 1, 100)
    k = 1 / sigma
    sz_pdf = (k**k * a**(k - 1) * np.exp(-k * a)) / gamma(k)
    return [N0 * a, sz_pdf]

def Gauss_PD(N0, sigma):
    a = np.linspace(max(1e-2 * N0, N0 - 5 * sigma), N0 + 5 * sigma, 100)
    return [a, Gauss(a, N0, sigma)]

def polymer_chains_excl_vol_cyl_xs_N_SZ(q, N, sigma, L0, L_per, R_xs, sigma_R_xs):
    N_grid, D = Shulz_Zimm(N, sigma)
    
    # 1. Expand L into a 2D array: Shape (nodes, 1)
    L = (L0 * N_grid)[:, np.newaxis]
    
    # 2. Pass the 1D q and 2D L arrays into S_full directly.
    # SciPy handles the broadcasting internally in C. The Python loop is eradicated.
    S_evaluated = S_full(q, L, L_per)

    weights = D * N_grid**2
    den = simpson(y=weights, x=N_grid)
    integrand = weights[:, np.newaxis] * S_evaluated
    nom = simpson(y=integrand, x=N_grid, axis=0)
    ff_poly = nom / den

    cyl = cylinder(q, R_xs, sigma_R_xs)
    return ff_poly * cyl

def kholodenko_worm_cyl_xs_N_SZ(q, N, sigma, L0, L_per, R_xs, sigma_R_xs):
    # This remains largely unchanged as jscatter ff evaluations might not support 2D L arrays directly
    N_grid, D = Shulz_Zimm(N, sigma)
    S_evaluated = np.zeros((len(N_grid), len(q)))
    for i, N_val in enumerate(N_grid):
        S_evaluated[i, :] = js.ff.wormlikeChain(q, L0 * N_val, L_per).Y

    weights = D * N_grid**2
    den = simpson(y=weights, x=N_grid)
    integrand = weights[:, np.newaxis] * S_evaluated
    nom = simpson(y=integrand, x=N_grid, axis=0)
    ff_poly = nom / den

    cyl = cylinder(q, R_xs, sigma_R_xs)
    return ff_poly * cyl
    
    
