import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy as sc
import jscatter as js
import scipy.stats as st
import copy
import time
import pickle
import multiprocessing
import scipy.integrate as integrate
from scipy.integrate import simpson
from glob import glob
import lmfit
from lmfit import Parameters, minimize, fit_report
import corner
from numba import njit

from EXCL_VOL_POL import polymer_chains_excl_vol_cyl_xs_N_SZ, kholodenko_worm_cyl_xs_N_SZ, cylinder, get_Rg_excl_vol, Shulz_Zimm, Gauss_PD
from ScatTools.sas import D11_SANS

# =============================================================================
# 1. GLOBAL CONSTANTS & MEMORY INITIALIZATION
# =============================================================================
SLD_PNIPAM = 0.67
SLD_PEG = 0.64
SLD_water = 6.388

M_auxillary = 2*(12*12+25 + 3*32) + 2*(7*12+14+32+9)
M_sc = 44*19+99
V0 = (62*19+80)*1e-3
V_molecule = (V0) * 576
sigma_in = 0.5
print(f"V_mol = {V0:.3f} nm^3")

# Load Datasets globally so that spawned workers natively inherit them via Linux CoW
list_of_files = glob('/mnt/data/Brush_SANS/D11_Brush/exp_9-11-2289_252_d11/processed/autostitch/*')
# NOTE: If directory orders change, replace these indices with exact string paths to avoid mismatches.
D2O_1mm = D11_SANS(list_of_files[12])
D2O_2mm = D11_SANS(list_of_files[16])

NIPAM_567_6_0p6 = D11_SANS(list_of_files[15])
NIPAM_567_6_0p6_40C = D11_SANS(list_of_files[18])
NIPAM_567_6_2p5 = D11_SANS(list_of_files[14])
NIPAM_567_6_2p5_40C = D11_SANS(list_of_files[17])
NIPAM_596_24_0p6 = D11_SANS(list_of_files[35])
NIPAM_596_24_0p6_40C = D11_SANS(list_of_files[37])
NIPAM_596_24_1p25 = D11_SANS(list_of_files[34])
NIPAM_596_24_1p25_40C = D11_SANS(list_of_files[36])

# Shared Beam Profile Generation
_meas = js.dataArray(np.array([D2O_2mm.q, D2O_2mm.I, D2O_2mm.I_err, D2O_2mm.q_err]))
_beam_profile = js.sas.prepareBeamProfile(_meas, explicit=3)

shared_data = {
    'SLD_water': SLD_water,
    'SLD_PEG': SLD_PEG,
    'SLD_PNIPAM': SLD_PNIPAM,
    'V_molecule': V_molecule,
    'beam_profile': _beam_profile,
}

# Quadrature Grids
_x, _w = np.polynomial.legendre.leggauss(40)
ALPHA_NODES = 0.5 * (_x + 1) * (np.pi / 2)
ALPHA_WEIGHTS = 0.5 * (np.pi / 2) * _w

low_ctf = 1
high_ctf = 190

# =============================================================================
# 2. OPTIMIZED MATHEMATICAL CORE
# =============================================================================
def get_N_agg_dist(N_mean, sigma_N, nodes=15):
    if sigma_N <= 1e-3:
        return np.array([N_mean]), np.array([1.0])
    N_min = max(10.0, N_mean - 3 * sigma_N)
    N_max = N_mean + 3 * sigma_N
    N_grid = np.linspace(N_min, N_max, nodes)
    weights = np.exp(-0.5 * ((N_grid - N_mean) / sigma_N)**2)
    weights /= np.sum(weights)
    return N_grid, weights

def psi_chains(q, N, sigma, L0, L_per, R_xs, sigma_R_xs):
    return np.sqrt(polymer_chains_excl_vol_cyl_xs_N_SZ(q, N, sigma, L0, L_per, R_xs, sigma_R_xs))

@njit(fastmath=True)
def ell_micelle_chains_numba(q_2d, alpha_2d, weights, R_core, eps_core, SLD_core, SLD_chain, 
                             SLD_water_val, V_molecule_val, sigma_in_val, N_agg, R_g_chain, d, alpha_dec, 
                             R_xs, a_chain, p_chain):
    """C-compiled core-shell cylinder form factor."""
    V_core = (4.0 / 3.0) * np.pi * eps_core * (R_core**3)
    
    d_rho_core = np.abs(SLD_core - SLD_water_val) * 1e-4        
    d_rho_chains = np.abs(SLD_chain - SLD_water_val) * 1e-4    
    
    M_core = (d_rho_core * V_core) * 1e-7            
    M_chains = (d_rho_chains * V_molecule_val) * 1e-7          

    r_core = R_core * np.sqrt(np.sin(alpha_2d)**2 + (eps_core**2) * np.cos(alpha_2d)**2)
    R_decorr = (4.0 * R_core**2) / (N_agg * R_xs)

    qr_core = q_2d * r_core
    qr_chain_pos = q_2d * (r_core + d * R_g_chain)
    
    # Inlined sphere form factor
    amp_core = (3.0 * (np.sin(qr_core) - qr_core * np.cos(qr_core)) / (qr_core**3)) * np.exp(-0.5 * (q_2d * sigma_in_val)**2)
    phase_chain = np.exp(-alpha_dec * q_2d * R_decorr) * np.sin(qr_chain_pos) / qr_chain_pos

    int_core = np.sum((amp_core**2) * np.sin(alpha_2d) * weights, axis=1)
    int_core_chain = np.sum(amp_core * phase_chain * np.sin(alpha_2d) * weights, axis=1)
    int_chain_chain = np.sum((phase_chain**2) * np.sin(alpha_2d) * weights, axis=1)
    
    ff_unnorm = ((M_core**2) * int_core
                 + N_agg * (M_chains**2) * p_chain
                 + 2.0 * N_agg * M_core * M_chains * a_chain * int_core_chain
                 + N_agg * (N_agg - 1.0) * (M_chains**2) * int_chain_chain * p_chain)
                 
    return ff_unnorm

def Brush_mic(params, q, intensity=None, eps=None):
    """Calculates Model Intensity. If experimental intensity & eps are provided, returns residuals."""
    if intensity is not None:
        plot = False
        parvals = params.valuesdict()
    else:
        plot = True
        # Allow passing either a raw lmfit.Parameters object or a MinimizerResult for plotting
        parvals = params.params.valuesdict() if hasattr(params, 'params') else params.valuesdict()

    phi_core = parvals['phi_core']
    N_agg_mean = parvals['N_agg']
    sigma_N_agg = parvals['sigma_N_agg']
    loops = parvals['loops']
    N_PNIPAM = parvals['N_PNIPAM']
    eps_core = parvals['eps_core']
    alpha_dec = parvals['alpha_dec']
    N_brush = parvals['N_brush'] / loops
    sigma_brush = parvals['sigma']
    L0 = parvals['L0']
    L_per = parvals['b']
    R_xs = parvals['R_xs']
    sigma_R_xs = parvals['sigma_R_xs']
    d = parvals['d']
    sigma_in_val = parvals['sigma_in']
    conc = parvals['conc']   
    bckg = parvals['bckg']
    
    SLD_water_val = shared_data['SLD_water']
    SLD_chain = shared_data['SLD_PEG']
    beam_profile = shared_data['beam_profile']
    
    SLD_core = shared_data['SLD_PNIPAM'] * (1 - phi_core) + SLD_water_val * phi_core
    V_PNIPAM = (N_PNIPAM * 187.6) * 1e-3
    
    # Calculate N-dependent molecular volume scalar mapped to V_molecule in Numba
    V_molecule_val = V0 * N_brush
    
    R_g_chain = get_Rg_excl_vol(N_brush * L0, L_per)
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
        
        P_q_n = ell_micelle_chains_numba(
            q_2d, alpha_2d, weights, R_core_n, eps_core, SLD_core, SLD_chain, 
            SLD_water_val, V_molecule_val, sigma_in_val, n_agg * loops, R_g_chain, d, alpha_dec, 
            R_xs, a_chain, p_chain
        )
        P_q_mean += np.nan_to_num(P_q_n, nan=0.0) * w

    conc_mM = 1e3 * conc / ((loops * N_brush * M_sc) + 2 * N_PNIPAM * 113 + M_auxillary)
    N_density = (conc_mM * 1e-6) * sc.constants.N_A / N_agg_mean
    
    H0 = (L0)**(3/5) * N_brush**(3/5) * (2*R_xs)**(2/5) * (N_agg_mean * loops)**(1/5)
    R_eff = R_core_mean + H0
    V_eff = (4/3) * np.pi * (R_eff)**3
    eta = N_density * (V_eff * 1e-21)
    
    try:
        stf = js.sf.PercusYevick(q, R_eff, eta=eta)
    except Exception:
        stf = js.sf.PercusYevick(q, 12, eta=0.09)

    calc = js.dataArray(np.array([q, P_q_mean]))
    
    unsmeared = N_density * calc * stf + bckg
    smeared_calc = js.sas.smear(unsmeared=unsmeared, beamProfile=beam_profile).Y

    if plot:
        S_core = 4 * np.pi * R_core_mean**2
        S_chain = np.pi * R_xs**2
        print(f"R_mean = {R_core_mean:.3f} nm; R_eff = {R_eff:.3f} nm; eta = {eta:.3f}")
        print(f"S_bb_agg = {S_core/(loops*N_agg_mean):.3f} nm^2; S_bb = {S_chain:.3f} nm^2; ratio = {S_core/(loops*N_agg_mean)/S_chain:.2f}")
        print(f"R_g_chain = {R_g_chain:.3f} nm ")
        print(f"H_0 = {H0:.3f} nm")
        return smeared_calc
    else:
        return (intensity - smeared_calc) / eps

def make_plot(data, fitted_pars):
    """Generates and saves visual fit reports."""
    fig, [ax, ax1] = plt.subplots(2, 1, figsize=(6, 8), layout='tight', height_ratios=[5, 1])
    bckg = fitted_pars.params['bckg'].value
    
    q_data = data.q[low_ctf:high_ctf]
    i_data = data.I[low_ctf:high_ctf]
    i_err = data.I_err[low_ctf:high_ctf]
    
    ax.errorbar(q_data, i_data - bckg, yerr=i_err, fmt='o', zorder=0, alpha=0.3, capsize=3, label='Experimental Data')
    ax.loglog(q_data, Brush_mic(fitted_pars, q_data) - bckg, '-', linewidth=2.5, zorder=2, label='Analytical Fit')
    
    ax.set_xlabel('$q \\ \\mathrm{[nm^{-1}]}$', fontsize=12)
    ax.set_ylabel('$\\Delta\\Sigma / \\Delta\\Omega \\ \\mathrm{[cm^{-1}]}$', fontsize=12)
    ax.set_ylim(1e-5, max(i_data)*1.5)
    ax.legend()
    ax.set_title(data.sample_name, fontsize=14)
    
    ax1.semilogx(q_data, fitted_pars.residual, '.', alpha=0.6)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('$q \\ \\mathrm{[nm^{-1}]}$', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    
    _lim = 1.5 * max(max(fitted_pars.residual), -min(fitted_pars.residual))
    ax1.set_ylim(-_lim, _lim)
    
    plt.savefig(f"Fit_{data.sample_name}.png", dpi=300)
    plt.close(fig)

# =============================================================================
# 3. BATCH EXECUTION FRAMEWORK
# =============================================================================

def build_parameters(config_dict):
    """Helper function to dynamically build lmfit Parameters from config dict."""
    pars = Parameters()
    for p_name, p_args in config_dict.items():
        if isinstance(p_args, tuple):
            val, vmin, vmax, vary = p_args
            pars.add(p_name, value=val, min=vmin, max=vmax, vary=vary)
        else:
            pars.add(p_name, value=p_args, vary=False)
    return pars

if __name__ == '__main__':
    
    # Define execution pipeline for the 4 specific datasets
    # Tuples: (value, min, max, vary_boolean)
    batch_configs = [
        {
            'name': 'NIPAM_596_24_0p6_40C',
            'raw_data': NIPAM_596_24_0p6_40C,
            'bckg_data': D2O_2mm,
            'mcmc_steps': 6400,
            'params': {
                'N_agg':      (40, 5, 150, True),
                'sigma_N_agg':(5, 0, 25, False),
                'loops':      (1, 1, 2, False),
                'd':          (0.4, 0.05, 1.5, True),
                'phi_core':   (0.2, 0, 0.95, True),
                'eps_core':   (1, 1, 4, False),
                'alpha_dec':  (1, 0.1, 100, True),
                'N_PNIPAM':   632,
                'N_brush':    576,
                'sigma':      0.1,
                'L0':         (0.18, 0.1, 0.254, True),
                'b':          (21.3, 5, 200, True),
                'R_xs':       (2.8, 0.5, 40, True),
                'sigma_R_xs': (0.5, 0.1, 4, True),
                'sigma_in':   2,
                'bckg':       (0.00346, 1e-7, 0.1, True),
                'conc':       0.625,
                'SLD_brush':  SLD_PEG
            }
        },
        {
            'name': 'NIPAM_596_24_1p25_40C',
            'raw_data': NIPAM_596_24_1p25_40C,
            'bckg_data': D2O_1mm,
            'mcmc_steps': 6400,
            'params': {
                'N_agg':      (40, 15, 150, True),
                'sigma_N_agg':(5, 0, 25, False),
                'loops':      (1, 1, 2, False),
                'd':          (0.8, 0.05, 1.2, True),
                'phi_core':   (0.2, 0, 0.95, True),
                'eps_core':   (1, 1, 2, False),
                'alpha_dec':  (1, 0.1, 100, True),
                'N_PNIPAM':   632,
                'N_brush':    576,
                'sigma':      0.1,
                'L0':         (0.18, 0.1, 0.254, True),
                'b':          (21.3, 5, 100, True),
                'R_xs':       (2.8, 0.5, 40, True),
                'sigma_R_xs': (0.5, 0.1, 4, True),
                'sigma_in':   1,
                'bckg':       (0.00546, 1e-7, 0.1, True),
                'conc':       1.25,
                'SLD_brush':  SLD_PEG
            }
        },
        {
            'name': 'NIPAM_567_6_0p6_40C',
            'raw_data': NIPAM_567_6_0p6_40C,
            'bckg_data': D2O_2mm,
            'mcmc_steps': 6400,
            'params': {
                'N_agg':      (40, 5, 150, True),
                'sigma_N_agg':(5, 0, 20, False),
                'loops':      (1, 1, 2, False),
                'd':          (0.8, 0.01, 2, True),
                'phi_core':   (0.7, 0, 0.95, True),
                'eps_core':   (1, 1, 2, False),
                'alpha_dec':  (1, 0, 15, True),
                'N_PNIPAM':   189,
                'N_brush':    576,
                'sigma':      0.1,
                'L0':         (0.18, 0.02, 0.254, True),
                'b':          (44, 5, 100, True),
                'R_xs':       (2.7, 0.5, 4, True),
                'sigma_R_xs': (0.6, 0.1, 4, True),
                'sigma_in':   2,
                'bckg':       (0.004, 1e-7, 0.1, True),
                'conc':       0.625,
                'SLD_brush':  SLD_PEG
            }
        },
        {
            'name': 'NIPAM_567_6_2p5_40C',
            'raw_data': NIPAM_567_6_2p5_40C,
            'bckg_data': D2O_1mm,
            'mcmc_steps': 6400,
            'params': {
                'N_agg':      (40, 5, 150, True),
                'sigma_N_agg':(5, 0, 20, False),
                'loops':      (1, 1, 2, False),
                'd':          (0.8, 0.01, 2, True),
                'phi_core':   (0.7, 0, 0.95, True),
                'eps_core':   (1, 1, 2, False),
                'alpha_dec':  (1, 0, 15, True),
                'N_PNIPAM':   189,
                'N_brush':    576,
                'sigma':      0.1,
                'L0':         (0.18, 0.02, 0.254, True),
                'b':          (44, 5, 100, True),
                'R_xs':       (2.7, 0.5, 4, True),
                'sigma_R_xs': (0.6, 0.1, 4, True),
                'sigma_in':   2,
                'bckg':       (0.004, 1e-7, 0.1, True),
                'conc':       2.5,
                'SLD_brush':  SLD_PEG
            }
        }
    ]

    for job in batch_configs:
        print(f"\n{'='*60}")
        print(f"Starting Analysis Pipeline for: {job['name']}")
        print(f"{'='*60}")
        
        # Setup specific data arrays and parameter dictionary
        data = copy.deepcopy(job['raw_data'])
        data.I = job['raw_data'].I - 1 * job['bckg_data'].I
        
        q_fit = data.q[low_ctf:high_ctf]
        i_fit = data.I[low_ctf:high_ctf]
        err_fit = data.I_err[low_ctf:high_ctf]
        
        pars = build_parameters(job['params'])
        
        # ---------------------------------------------------------
        # PHASE 1: Levenberg-Marquardt Determinstic Minimization
        # ---------------------------------------------------------
        print("--> Executing Least-Squares pre-fit...")
        t0 = time.time()
        fitted_pars = minimize(Brush_mic, pars, args=(q_fit, i_fit, err_fit), method='least_squares')  
        print(f"Least-Squares Fitting took {(time.time()-t0):.2f} s ({(time.time()-t0)/fitted_pars.nfev:.4f} s/it)")
        
        # Output artifacts
        make_plot(data, fitted_pars)
        with open(f"fit_params_{data.sample_name}.dat", 'w') as f:
            f.write(str(fit_report(fitted_pars)))
            
        with open(f"{data.sample_name}_fit.pkl", 'wb') as f:
            pickle.dump(fitted_pars, f)

        # ---------------------------------------------------------
        # PHASE 2: Parallelized MCMC emcee Sampling
        # ---------------------------------------------------------
        print(f"--> Initializing MCMC Sampling (Steps: {job['mcmc_steps']}) across {multiprocessing.cpu_count()} cores...")
        
        res = minimize(
            Brush_mic, 
            args=(q_fit, i_fit, err_fit), 
            method='emcee', 
            nan_policy='omit', 
            burn=1000, 
            steps=job['mcmc_steps'], 
            thin=20, 
            params=fitted_pars.params, 
            is_weighted=True, 
            progress=True,
            workers=multiprocessing.cpu_count()  # Built-in Pool generation handled by emcee
        )
        
        print("--> Generating Corner Plot...")
        truths = res.params.valuesdict()
        
        # Pop fixed/static values from Truths to avoid errors in corner plots
        fixed_keys = ['sigma_N_agg', 'loops', 'eps_core', 'N_PNIPAM', 'N_brush', 'sigma', 'sigma_in', 'SLD_brush', 'conc']
        for key in fixed_keys:
            truths.pop(key, None)

        emcee_plot = corner.corner(res.flatchain, labels=res.var_names, truths=list(truths.values()))
        emcee_plot.savefig(f"{data.sample_name}_MCMC_out.png", dpi=300)
        plt.close(emcee_plot)
        
        with open(f"fit_params_{data.sample_name}_MCMC.dat", 'w') as f:
            f.write(str(fit_report(res)))
            
        print(f"Completed {job['name']}.")