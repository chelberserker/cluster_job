### Method 1 with excluded volume contribution
### Macromolecules 29, 23, 7602-7612, 1996

import numpy as np
import jscatter as js
from scipy.special import jv, sici, gamma
from scipy.differentiate import derivative
from scipy.integrate import  simpson
from chi_coefs import *

def Gauss(R, R0=50, sigma=5):
    """
    Calculation of normalized gaussian
    Args:
        R (float):
        sigma (float):
    Returns:
        np.ndarray: Gaussian distribution centered at R with dispersion sigma
    """
    G = (1/((2*np.pi)**0.5*sigma))*np.e**(-0.5*((R-R0)/sigma)**2)
    return G

def Si(x):
    """
    Wrapper function for Sinc(t)= Int_o^t sin(x)/x dx
    Args:
        x (np.ndarray): upper integration limit
    Returns:
        np.ndarray: Sinc(t)= Int_o^t sin(x)/x dx
    """
    res = sici(x)[0]
    return res

def S_rod(q, L):
    """
    Calculates scattering of a rigid rod
    :param q:
    :param L:
    :return:
    """
    res = 2*Si(q*L)/(q*L) - 4*(np.sin(q*L/2))**2/(q**2*L**2)
    return res

def gauss_chain(q, R_g):
    x = R_g**2*q**2
    P = (2*(np.exp(-x) - 1 + x)/(x**2))
    return P

def cylinder(q, R0, sigmaR):
    R = np.linspace(max(1e-2*R0, R0-5*sigmaR), R0+5*sigmaR, 500)
    rad_dist = Gauss(R, R0, sigmaR)
    qR = q[:, np.newaxis]*R
    with np.errstate(divide='ignore', invalid='ignore'):
        form_factor = (2 * jv(1, qR) / qR) ** 2
        # If q=0 is present, fix the NaN singularity (the limit as x->0 of 2*J1(x)/x is 1)
        form_factor = np.where(qR == 0, 1.0, form_factor)

    #Int = trapezoid(form_factor * rad_dist, x=R, axis=1)
    Int = np.sum(form_factor * rad_dist, axis=1)
    res = Int / np.max(Int)
    return res


def get_Rg_excl_vol(L,b):
    n_b = L/b
    R_g0_2 = ((L*b)/6)
    alpha_2 = (1 + (n_b/3.12)**2 + (n_b/8.67)**3)**(0.170/3)
    R_g = np.sqrt(R_g0_2 * alpha_2)
    return R_g

def xi_func(q, L, b):         #equation 17
    R_g = get_Rg_excl_vol(L, b)
    xi = q*b*(np.pi*b/(1.103*L))**(3/2)*(R_g**2/b**2)**1.282
    return xi

def chi_func(q, L, b):
    xi = xi_func(q, L, b)
    chi = np.exp(-xi**(-5))
    return chi

def Gamma_func(q, L, b):
    chi = chi_func(q, L, b)
    sum1 = 0
    for i in range(2,6):
        sum1 += get_Ai(i, L, b)*xi_func(q, L, b)**i
    sum2 = 0
    for i in range(0,3):
        sum2 += get_Bi(i, L, b)*xi_func(q, L, b)**(-i)

    gam_func = 1 + (1-chi)*sum1 + chi*sum2

    return gam_func


def excl_vol(q, L, b):
    # n_b = L/b
    # R_g0_2 = ((L*b)/6)
    # alpha_2 = (1 + (n_b/3.12)**2 + (n_b/8.67)**3)**(0.170/3)
    # R_g = np.sqrt(R_g0_2 * alpha_2)
    R_g = get_Rg_excl_vol(L, b)
    # print(R_g)

    C1 = 1.220
    C2 = 0.4288
    C3 = -1.651
    C4 = 1.523
    C5 = 0.1477
    nu = 0.588

    w = (1 + np.tanh((q * R_g - C4) / C5)) / 2
    sum1 = C1 * (q * R_g) ** (-1 / nu)
    sum2 = C2 * (q * R_g) ** (-2 / nu)
    sum3 = C3 * (q * R_g) ** (-3 / nu)

    result = (1 - w) * gauss_chain(q, R_g) + w * (sum1 + sum2 + sum3)

    return result


def self_av_chain(q, L, b):
    p1 = 4.12
    p2 = 4.42
    p3 = -0.44
    a4 = 3.06

    S_excl_vol = excl_vol(q, L, b)
    R_g = get_Rg_excl_vol(L, b)
    if (L > 10 * b):
        C = a4 / ((L / b) ** p3)
    else:
        C = 1

    u = q ** 2 * R_g ** 2

    mult1 = 4 / 15 + 7 / (15 * u) - (11 / 15 + (7 / (15 * u))) * np.exp(-u)

    S_chain = S_excl_vol + C * mult1 * (b / L)

    # S_chain = gauss_chain(q, R_g) + mult1*(b/L)

    # xi = q*b * (np.pi*b/(1.103*L))**(3/2) * ((R_g/b)**2)*1.282
    # chi = np.exp(-xi**(-5))

    # S_full

    return S_chain


def S_full(q, L, b):
    S_c = excl_vol(q, L, b)
    chi = chi_func(q, L, b)
    gamma = Gamma_func(q, L, b)
    rod = S_rod(q, L)
    S_wc = ((1-chi)*S_c + chi*rod)*gamma

    return S_wc


def polymer_chains_excl_vol(q, L, b):
    chains = S_full(q, L, b)
    return chains

def polymer_chains_excl_vol_cyl_xs(q, L, b, R_xs, sigma_R_xs):
    chains = S_full(q, L, b)
    cyl = cylinder(q, R_xs, sigma_R_xs)

    res = chains * cyl
    return res

def Shulz_Zimm(N0, sigma):
    _a_min = -np.log10(N0)
    a = np.logspace(_a_min, 1, 400)
    k = 1/sigma
    sz_pdf = (k**k*a**(k-1)*np.exp(-k*a))/(gamma(k))

    return [N0*a, sz_pdf]

def Gauss_PD(N0, sigma):
    a = np.linspace(max(1e-2*N0, N0-5*sigma), N0+5*sigma, 400)
    gauss_pdf = Gauss(a, N0, sigma)
    return [a, gauss_pdf]

def polymer_chains_excl_vol_cyl_xs_N_SZ(q, N, sigma, L0, L_per, R_xs, sigma_R_xs):
    N_grid, D = Shulz_Zimm(N, sigma)
    S_evaluated = np.zeros((len(N_grid), len(q)))
    for i, N in enumerate(N_grid):
        S_evaluated[i, :] = polymer_chains_excl_vol(q, L0 * N, L_per)

    weights = D * N_grid**2
    den = simpson(y=weights, x=N_grid)
    integrand = weights[:, np.newaxis] * S_evaluated
    nom = simpson(y=integrand, x=N_grid, axis=0)
    ff_poly = nom / den

    cyl = cylinder(q, R_xs, sigma_R_xs)
    res = ff_poly*cyl    
    
    return res

def kholodenko_worm_cyl_xs_N_SZ(q, N, sigma, L0, L_per, R_xs, sigma_R_xs):
    N_grid, D = Shulz_Zimm(N, sigma)
    S_evaluated = np.zeros((len(N_grid), len(q)))
    for i, N in enumerate(N_grid):
        S_evaluated[i, :] = js.ff.wormlikeChain(q, L0 * N, L_per).Y

    weights = D * N_grid**2
    den = simpson(y=weights, x=N_grid)
    integrand = weights[:, np.newaxis] * S_evaluated
    nom = simpson(y=integrand, x=N_grid, axis=0)
    ff_poly = nom / den

    cyl = cylinder(q, R_xs, sigma_R_xs)
    res = ff_poly*cyl    
    
    return res
