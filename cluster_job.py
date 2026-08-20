import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy as sc
import jscatter as js
import scipy.stats as st
import copy
import time
import multiprocessing
import scipy.integrate as integrate
from scipy.integrate import simpson
from glob import glob
import lmfit
from lmfit import Parameters, minimize, fit_report
import corner


import cProfile
import pstats

from numba import njit

from EXCL_VOL_POL import polymer_chains_excl_vol_cyl_xs_N_SZ, kholodenko_worm_cyl_xs_N_SZ, cylinder, get_Rg_excl_vol, Shulz_Zimm, Gauss_PD
from ScatTools.sas import D11_SANS

SLD_PNIPAM = 0.67
SLD_PEG = 0.64
SLD_water = 6.388

M_auxillary = 2*(12*12+25 + 3*32) + 2*(7*12+14+32+9)
M_sc = 44*19+99
V_0 = (62*19+80)*1e-3
print(f"V_mol = {V_0:.3f} nm^3")

V_molecule = (V_0) * 576
sigma_in = 0.5

# Global dictionary initialized for Linux Copy-on-Write sharing
_D2O_2mm = D11_SANS('./IQ072126_c16.dat')
_meas = js.dataArray(np.array([_D2O_2mm.q, _D2O_2mm.I, _D2O_2mm.I_err, _D2O_2mm.q_err]))
_beam_profile = js.sas.prepareBeamProfile(_meas, explicit=3)

shared_data = {
    'SLD_water': SLD_water,
    'SLD_PEG': SLD_PEG,
    'SLD_PNIPAM': SLD_PNIPAM,
    'V_molecule': V_molecule,
    'beam_profile': _beam_profile,
}

_x, _w = np.polynomial.legendre.leggauss(40)
ALPHA_NODES = 0.5 * (_x + 1) * (np.pi / 2)
ALPHA_WEIGHTS = 0.5 * (np.pi / 2) * _w

def PD_worm_N_weigth(N, sigma, V0):
    N_grid, D_L = Shulz_Zimm(N, sigma)
    weights = D_L
    num = simpson(y=N_grid*V0*weights, x=N_grid)
    den = simpson(y=weights, x=N_grid)
    res = num/den
    return res

def ff_sphere(qR):
    return np.where(qR < 1e-6, 1.0, 3 * (np.sin(qR) - qR * np.cos(qR)) / (qR**3))

def psi_chains(q, N, sigma, L0, L_per, R_xs, sigma_R_xs):
    Amp = np.sqrt(polymer_chains_excl_vol_cyl_xs_N_SZ(q, N, sigma, L0, L_per, R_xs, sigma_R_xs))
    return Amp

def get_N_agg_dist(N_mean, sigma_N, nodes=31):
    if sigma_N <= 1e-3:
        return np.array([N_mean]), np.array([1.0])
    
    N_min = max(10.0, N_mean - 3 * sigma_N)
    N_max = N_mean + 3 * sigma_N
    
    N_grid = np.linspace(N_min, N_max, nodes)
    weights = np.exp(-0.5 * ((N_grid - N_mean) / sigma_N)**2)
    weights /= np.sum(weights)
    
    return N_grid, weights

def ell_micelle_chains(q, R_core, eps_core, SLD_core, SLD_chain, N, sigma, L0, L_per, R_xs, sigma_R_xs, N_agg, R_g_chain, d, alpha_dec):
    V_core = (4/3)*np.pi*eps_core*(R_core**3)
    chain_amp = psi_chains(q, N, sigma, L0, L_per, R_xs, sigma_R_xs)
    
    # Access shared_data directly (inherited by worker processes)
    SLD_water = shared_data['SLD_water']
    V_molecule = shared_data['V_molecule']
    
    d_rho_core = abs(SLD_core-SLD_water)*1e-4       
    d_rho_chains = abs(SLD_chain-SLD_water)*1e-4    
    
    M_core = (d_rho_core * V_core)*1e-7           
    M_chains = (d_rho_chains * V_molecule)*1e-7          
    M_mic = M_core + N_agg*M_chains

    R_g = get_Rg_excl_vol(N*L0, L_per)

    q_2d = q[:, np.newaxis]
    alpha_2d = ALPHA_NODES[np.newaxis, :]
    weights = ALPHA_WEIGHTS[np.newaxis, :]

    r_core = R_core * np.sqrt(np.sin(alpha_2d)**2 + (eps_core**2) * np.cos(alpha_2d)**2)

    qr_core = q_2d * r_core
    qr_chain_pos = q_2d * (r_core + d * R_g)
    
    amp_core = ff_sphere(qr_core) * np.exp(-0.5 * (q_2d * sigma_in)**2)

    R_decorr = (4*R_core**2)/(N_agg*R_xs)    
    
    phase_chain = np.where(qr_chain_pos < 1e-6, 1.0, np.exp(-alpha_dec*q_2d*R_decorr)*np.sin(qr_chain_pos) / qr_chain_pos)
    phase_chain /= phase_chain[0]

    a_chain = psi_chains(q, N, sigma, L0, L_per, R_xs, sigma_R_xs)
    p_chain = a_chain**2
    
    int_core = np.sum((amp_core**2) * np.sin(alpha_2d) * weights, axis=1)
    int_core_chain = np.sum(amp_core * phase_chain * np.sin(alpha_2d) * weights, axis=1)
    int_chain_chain = np.sum((np.exp(-2*q_2d*R_decorr)*phase_chain**2) * np.sin(alpha_2d) * weights, axis=1)
    
    ff = (M_mic**-2)*((M_core**2)*int_core
                      + N_agg*(M_chains**2)*p_chain
                      + 2*N_agg*M_core*M_chains*a_chain*int_core_chain
                      + N_agg*(N_agg-1)*(M_chains**2)*int_chain_chain*p_chain 
                     )
    ff = np.nan_to_num(ff, nan=1)
    
    return ff


def ell_micelle_chains_unnorm(q, R_core, eps_core, SLD_core, SLD_chain, N, sigma, L0, L_per, R_xs, sigma_R_xs, N_agg, R_g_chain, d, alpha_dec):
    V_core = (4/3)*np.pi*eps_core*(R_core**3)
    chain_amp = psi_chains(q, N, sigma, L0, L_per, R_xs, sigma_R_xs)
    
    # Access shared_data directly (inherited by worker processes)
    SLD_water = shared_data['SLD_water']
    V_molecule = shared_data['V_molecule']
    
    d_rho_core = abs(SLD_core-SLD_water)*1e-4        
    d_rho_chains = abs(SLD_chain-SLD_water)*1e-4    
    
    M_core = (d_rho_core * V_core)*1e-7            
    M_chains = (d_rho_chains * V_molecule)*1e-7          

    q_2d = q[:, np.newaxis]
    alpha_2d = ALPHA_NODES[np.newaxis, :]
    weights = ALPHA_WEIGHTS[np.newaxis, :]

    r_core = R_core * np.sqrt(np.sin(alpha_2d)**2 + (eps_core**2) * np.cos(alpha_2d)**2)

    R_decorr = (4*R_core**2)/(N_agg*R_xs)

    qr_core = q_2d * r_core
    qr_chain_pos = q_2d * (r_core + d * R_g_chain)
    
    amp_core = ff_sphere(qr_core) * np.exp(-0.5 * (q_2d * sigma_in)**2)

    phase_chain = np.where(qr_chain_pos < 1e-6, 1.0, np.exp(-alpha_dec*q_2d*R_decorr)*np.sin(qr_chain_pos) / qr_chain_pos)

    a_chain = psi_chains(q, N, sigma, L0, L_per, R_xs, sigma_R_xs)
    p_chain = a_chain**2
    
    int_core = np.sum((amp_core**2) * np.sin(alpha_2d) * weights, axis=1)
    int_core_chain = np.sum(amp_core * phase_chain * np.sin(alpha_2d) * weights, axis=1)
    int_chain_chain = np.sum((phase_chain**2) * np.sin(alpha_2d) * weights, axis=1)
    
    ff_unnorm = ((M_core**2)*int_core
                      + N_agg*(M_chains**2)*p_chain
                      + 2*N_agg*M_core*M_chains*a_chain*int_core_chain
                      + N_agg*(N_agg-1)*(M_chains**2)*int_chain_chain*p_chain 
                     )
                 
    return np.nan_to_num(ff_unnorm, nan=0.0)
    
def ell_micelle_chains_unnorm_opt(q_2d, alpha_2d, weights, R_core, eps_core, SLD_core, SLD_chain, 
                                  N_agg, R_g_chain, d, alpha_dec, R_xs, a_chain, p_chain):
    """Optimized core-shell cylinder form factor accepting pre-computed chain scattering."""
    V_core = (4/3)*np.pi*eps_core*(R_core**3)
    
    SLD_water = shared_data['SLD_water']
    V_molecule = shared_data['V_molecule']
    
    d_rho_core = abs(SLD_core-SLD_water)*1e-4        
    d_rho_chains = abs(SLD_chain-SLD_water)*1e-4    
    
    M_core = (d_rho_core * V_core)*1e-7            
    M_chains = (d_rho_chains * V_molecule)*1e-7          

    r_core = R_core * np.sqrt(np.sin(alpha_2d)**2 + (eps_core**2) * np.cos(alpha_2d)**2)
    R_decorr = (4*R_core**2)/(N_agg*R_xs)

    qr_core = q_2d * r_core
    qr_chain_pos = q_2d * (r_core + d * R_g_chain)
    
    # Inlined sphere form factor to remove function call overhead
    amp_core = (3 * (np.sin(qr_core) - qr_core * np.cos(qr_core)) / (qr_core**3)) * np.exp(-0.5 * (q_2d * sigma_in)**2)

    # Removed np.where; assuming q > 0 for standard SANS datasets
    phase_chain = np.exp(-alpha_dec*q_2d*R_decorr) * np.sin(qr_chain_pos) / qr_chain_pos

    int_core = np.sum((amp_core**2) * np.sin(alpha_2d) * weights, axis=1)
    int_core_chain = np.sum(amp_core * phase_chain * np.sin(alpha_2d) * weights, axis=1)
    int_chain_chain = np.sum((phase_chain**2) * np.sin(alpha_2d) * weights, axis=1)
    
    ff_unnorm = ((M_core**2)*int_core
                 + N_agg*(M_chains**2)*p_chain
                 + 2*N_agg*M_core*M_chains*a_chain*int_core_chain
                 + N_agg*(N_agg-1)*(M_chains**2)*int_chain_chain*p_chain)
                 
    return np.nan_to_num(ff_unnorm, nan=0.0)
    

@njit(fastmath=True)
def ell_micelle_chains_numba(q_2d, alpha_2d, weights, R_core, eps_core, SLD_core, SLD_chain, 
                             SLD_water, V_molecule, sigma_in, N_agg, R_g_chain, d, alpha_dec, 
                             R_xs, a_chain, p_chain):
    """
    Compiled C-level form factor execution. 
    fastmath=True relaxes IEEE-754 strictness for faster floating-point operations.
    """
    V_core = (4.0 / 3.0) * np.pi * eps_core * (R_core**3)
    
    d_rho_core = np.abs(SLD_core - SLD_water) * 1e-4        
    d_rho_chains = np.abs(SLD_chain - SLD_water) * 1e-4    
    
    M_core = (d_rho_core * V_core) * 1e-7            
    M_chains = (d_rho_chains * V_molecule) * 1e-7          

    r_core = R_core * np.sqrt(np.sin(alpha_2d)**2 + (eps_core**2) * np.cos(alpha_2d)**2)
    R_decorr = (4.0 * R_core**2) / (N_agg * R_xs)

    qr_core = q_2d * r_core
    qr_chain_pos = q_2d * (r_core + d * R_g_chain)
    
    # Inlined sphere form factor
    amp_core = (3.0 * (np.sin(qr_core) - qr_core * np.cos(qr_core)) / (qr_core**3)) * np.exp(-0.5 * (q_2d * sigma_in)**2)
    phase_chain = np.exp(-alpha_dec * q_2d * R_decorr) * np.sin(qr_chain_pos) / qr_chain_pos

    int_core = np.sum((amp_core**2) * np.sin(alpha_2d) * weights, axis=1)
    int_core_chain = np.sum(amp_core * phase_chain * np.sin(alpha_2d) * weights, axis=1)
    int_chain_chain = np.sum((phase_chain**2) * np.sin(alpha_2d) * weights, axis=1)
    
    ff_unnorm = ((M_core**2) * int_core
                 + N_agg * (M_chains**2) * p_chain
                 + 2.0 * N_agg * M_core * M_chains * a_chain * int_core_chain
                 + N_agg * (N_agg - 1.0) * (M_chains**2) * int_chain_chain * p_chain)
                 
    return ff_unnorm

def Brush_mic(params, q, intensity, eps=None):
    parvals = params.valuesdict()
    
    phi_core = parvals['phi_core']
    N_agg_mean = parvals['N_agg']
    sigma_N_agg = parvals['sigma_N_agg']
    N_PNIPAM = parvals['N_PNIPAM']
    eps_core = parvals['eps_core']
    alpha_dec = parvals['alpha_dec']
    N_brush = parvals['N_brush']
    sigma_brush = parvals['sigma']
    L0 = parvals['L0']
    L_per = parvals['b']
    R_xs = parvals['R_xs']
    sigma_R_xs = parvals['sigma_R_xs']
    loops = parvals['loops']
    d = parvals['d']
    conc = parvals['conc']   
    bckg = parvals['bckg']
    
    SLD_water_val = shared_data['SLD_water']
    V_molecule_val = shared_data['V_molecule']
    SLD_chain = shared_data['SLD_PEG']
    beam_profile = shared_data['beam_profile']
    
    SLD_core = SLD_PNIPAM * (1 - phi_core) + SLD_water_val * phi_core
    V_PNIPAM = (N_PNIPAM*187.6)*1e-3
    
    R_g_chain = get_Rg_excl_vol(N_brush*L0, L_per)
    
    N_grid, N_weights = get_N_agg_dist(N_agg_mean, sigma_N_agg, nodes=15)

    a_chain = psi_chains(q, N_brush, sigma_brush, L0, L_per, R_xs, sigma_R_xs)
    p_chain = a_chain**2

    q_2d = q[:, np.newaxis]
    alpha_2d = ALPHA_NODES[np.newaxis, :]
    weights = ALPHA_WEIGHTS[np.newaxis, :]

    P_q_mean = np.zeros_like(q)
    R_core_mean = 0.0

    for n_agg, w in zip(N_grid, N_weights):
        V_core_n = (loops * n_agg * V_PNIPAM) / (1 - phi_core)
        R_core_n = ((3 * V_core_n) / (4 * np.pi * eps_core))**(1/3)
        R_core_mean += R_core_n * w
        
        # Execute the Numba compiled function
        P_q_n = ell_micelle_chains_numba(
            q_2d, alpha_2d, weights, R_core_n, eps_core, SLD_core, SLD_chain, 
            SLD_water_val, V_molecule_val, sigma_in, n_agg, R_g_chain, d, alpha_dec, 
            R_xs, a_chain, p_chain
        )
        
        P_q_mean += np.nan_to_num(P_q_n, nan=0.0) * w
        
    # Use the pre-calculated weight array
    conc_mM = 1e3 * conc / (N_brush*M_sc + 2*N_PNIPAM*113 + M_auxillary)
    N_density = (conc_mM*1e-6)*sc.constants.N_A / N_agg_mean
    
    R_eff = (R_core_mean*(2+eps_core))/3 + 2*R_g_chain
    V_eff = (4/3)*np.pi*(R_eff)**3
    eta = N_density*(V_eff*1e-21)
    
    try:
        stf = js.sf.PercusYevick(q, R_eff, eta=eta)
    except Exception:
        stf = js.sf.PercusYevick(q, 12, eta=0.09)

    calc = js.dataArray(np.array([q, P_q_mean]))
    
    unsmeared = N_density * calc * stf + bckg
    smeared_calc = js.sas.smear(unsmeared=unsmeared, beamProfile=beam_profile).Y    
    
    return (intensity - smeared_calc) / eps
    
        
def plot_Brush_mic(pars_fit, q):
    parvals = pars_fit.params.valuesdict()
    
    phi_core = parvals['phi_core']
    N_agg_mean = parvals['N_agg']
    sigma_N_agg = parvals['sigma_N_agg']
    N_PNIPAM = parvals['N_PNIPAM']
    eps_core = parvals['eps_core']
    alpha_dec = parvals['alpha_dec']
    N_brush = parvals['N_brush']
    sigma_brush = parvals['sigma']
    L0 = parvals['L0']
    L_per = parvals['b']
    R_xs = parvals['R_xs']
    sigma_R_xs = parvals['sigma_R_xs']
    loops = parvals['loops']
    d = parvals['d']
    conc = parvals['conc']   
    bckg = parvals['bckg']
    
    SLD_water_val = shared_data['SLD_water']
    V_molecule_val = shared_data['V_molecule']
    SLD_chain = shared_data['SLD_PEG']
    beam_profile = shared_data['beam_profile']
    
    SLD_core = SLD_PNIPAM * (1 - phi_core) + SLD_water_val * phi_core
    
    V_PNIPAM = (N_PNIPAM*187.6)*1e-3
    R_g_chain = get_Rg_excl_vol(N_brush*L0, L_per)
    N_grid, N_weights = get_N_agg_dist(N_agg_mean, sigma_N_agg, nodes=15)

    a_chain = psi_chains(q, N_brush, sigma_brush, L0, L_per, R_xs, sigma_R_xs)
    p_chain = a_chain**2

    q_2d = q[:, np.newaxis]
    alpha_2d = ALPHA_NODES[np.newaxis, :]
    weights = ALPHA_WEIGHTS[np.newaxis, :]

    P_q_mean = np.zeros_like(q)
    R_core_mean = 0.0

    for n_agg, w in zip(N_grid, N_weights):
        V_core_n = (loops * n_agg * V_PNIPAM) / (1 - phi_core)
        R_core_n = ((3 * V_core_n) / (4 * np.pi * eps_core))**(1/3)
        R_core_mean += R_core_n * w
        
        # Execute the Numba compiled function
        P_q_n = ell_micelle_chains_numba(
            q_2d, alpha_2d, weights, R_core_n, eps_core, SLD_core, SLD_chain, 
            SLD_water_val, V_molecule_val, sigma_in, n_agg, R_g_chain, d, alpha_dec, 
            R_xs, a_chain, p_chain
        )
        
        P_q_mean += np.nan_to_num(P_q_n, nan=0.0) * w
        
    # Use the pre-calculated weight array
    conc_mM = 1e3 * conc / (N_brush*M_sc + 2*N_PNIPAM*113 + M_auxillary)
    N_density = (conc_mM*1e-6)*sc.constants.N_A / N_agg_mean
    
    R_eff = (R_core_mean*(2+eps_core))/3 + 2*R_g_chain
    V_eff = (4/3)*np.pi*(R_eff)**3
    eta = N_density*(V_eff*1e-21)
    
    try:
        stf = js.sf.PercusYevick(q, R_eff, eta=eta)
    except Exception:
        stf = js.sf.PercusYevick(q, 12, eta=0.09)

    calc = js.dataArray(np.array([q, P_q_mean]))
    
    unsmeared = N_density * calc * stf + bckg
    smeared_calc = js.sas.smear(unsmeared=unsmeared, beamProfile=beam_profile).Y    
        
    S_core = 4*np.pi*R_core_mean**2
    S_chain = np.pi*R_xs**2
    print(f"R_mean = {R_core_mean:.3f} nm; R_eff = {R_eff:.3f} nm; eta = {eta:.3f}")
    print(f"S_bb_agg = {S_core/(N_agg_mean*loops):.3f} nm^2; S_bb = {S_chain:.3f} nm^2; ratio = {S_core/(N_agg_mean*loops)/S_chain:.2f}")
    print(f"R_g_chain = {R_g_chain:.3f} nm ")
    return smeared_calc


if __name__ == '__main__':
    
    NIPAM_596_24_0p6_40C = D11_SANS('./IQ072237_c38.dat')
    
    np.random.seed(465865)
    data = copy.deepcopy(NIPAM_596_24_0p6_40C)  
    
    # Reference the globally loaded _D2O_2mm data for background subtraction
    data.I =  NIPAM_596_24_0p6_40C.I - 1 * _D2O_2mm.I  
    
    pars = Parameters()
    
    pars.add('N_agg', value = 20 , min=15, max=150, vary=True)
    pars.add('sigma_N_agg', value = 0 , min=0, max=25, vary=False)
    
    pars.add('phi_core', value = 0.50 , min=0, max=0.95, vary=True)
    pars.add('eps_core', value = 1, min=1, max=2,vary=False)
    pars.add('alpha_dec', value=1, min=0.1, max=1e2, vary=True)
    
    pars.add('N_PNIPAM', value = 632, min=50, max=700, vary=False)
    
    pars.add('N_brush', value = 576, min=50, max=700, vary=False)
    pars.add('sigma', value = 0.1, min=0.01, max=1, vary=False)
    pars.add('L0', value = 0.18, min=0.02, max=0.254, vary=True)
    
    pars.add('b', value = 21.3 , min=5, max=100, vary=True)
    pars.add('R_xs', value = 2.8 , min=0.5, max=40, vary=True)
    pars.add('loops', value=1, min=1, max=2, vary=False)
    pars.add('d', value=0.8, min=0.05, max=1.2, vary=True)
    pars.add('sigma_R_xs', value = 0.5 , min=0.1, max=4, vary=True)
    
    pars.add('bckg', value =  0.00346, min=1e-7, max=0.1, vary=True)
    pars.add('conc', value = 0.625 , min=1e-10, max=16, vary=False)
    pars.add('SLD_brush', value = SLD_PEG , min=0.1, max=0.8, vary=False)
    
    low_ctf = 1
    high_ctf = 190
    
    t0 = time.time()
    fitted_pars = minimize(Brush_mic, pars, args=(data.q[low_ctf:high_ctf], data.I[low_ctf:high_ctf], data.I_err[low_ctf:high_ctf]), method='least_squares')  
    print(f"Fitting took {(time.time()-t0):.2f} s")
    
    profiler = cProfile.Profile()
    profiler.enable()
    # Run a single evaluation
    Brush_mic(fitted_pars.params, data.q[low_ctf:high_ctf], data.I[low_ctf:high_ctf], data.I_err[low_ctf:high_ctf])
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(10)  # Prints the 10 most expensive function calls
        
    fig, ax = plt.subplots(1,1)
        
    ax.errorbar(data.q[low_ctf:high_ctf], data.I[low_ctf:high_ctf], data.I_err[low_ctf:high_ctf])
    ax.loglog(data.q[low_ctf:high_ctf], plot_Brush_mic(fitted_pars, data.q[low_ctf:high_ctf]),'--',linewidth=2, zorder=2, label='Fit')
    ax.set_xlabel('$q, \\mathrm{nm}^{-1}$')
    ax.set_ylabel('$\\Delta\\Omega, \\mathrm{cm}^{-1}$')
    plt.show()
    
    with open(f"fit_params_{data.sample_name}.dat", 'w') as f:
        f.write(str(fit_report(fitted_pars)))
    
    # emcee initialization passing the number of CPU cores directly
    res = minimize(
        Brush_mic, 
        args=(data.q[low_ctf:high_ctf], data.I[low_ctf:high_ctf], data.I_err[low_ctf:high_ctf]), 
        method='emcee', 
        nan_policy='omit', 
        burn=400, 
        steps=6400, 
        thin=1, 
        params=fitted_pars.params, 
        is_weighted=True, 
        progress=True, 
        workers=multiprocessing.cpu_count()
    )

    truths = res.params.valuesdict()
    truths.pop('SLD_brush')
    truths.pop('sigma_N_agg')
    truths.pop('eps_core')
    truths.pop('conc')
    truths.pop('N_PNIPAM')
    truths.pop('loops')
    truths.pop('sigma')
    truths.pop('N_brush')

    emcee_plot = corner.corner(res.flatchain, labels=res.var_names, truths=list(truths.values()))
    emcee_plot.savefig(f"{data.sample_name}_MCMC_out.png", dpi=300)
