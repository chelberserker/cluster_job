import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy as sc
import jscatter as js
import scipy.stats as st
import copy

from multiprocessing import Pool

import scipy.integrate as integrate
from scipy.integrate import simpson
from glob import glob
import lmfit
from lmfit import Parameters, minimize, fit_report
import corner

from EXCL_VOL_POL import polymer_chains_excl_vol_cyl_xs_N_SZ, kholodenko_worm_cyl_xs_N_SZ, cylinder, get_Rg_excl_vol, Shulz_Zimm, Gauss_PD
from ScatTools.sas import D11_SANS

SLD_PNIPAM = 0.67
SLD_PEG = 0.64
SLD_water = 6.388

#N_PNIPAM = 632  

M_auxillary = 2*(12*12+25 + 3*32) + 2*(7*12+14+32+9)
M_sc = 44*19+99
V_0 = (62*19+80)*1e-3
print(f"V_mol = {V_0:.3f} nm^3")

#V_PNIPAM = (N_PNIPAM * 187.6) * 1e-3 ## nm^3

V_molecule = (V_0) * 576


sigma_in = 0.5
#d=0.6


_x, _w = np.polynomial.legendre.leggauss(90)
ALPHA_NODES = 0.5 * (_x + 1) * (np.pi / 2)
ALPHA_WEIGHTS = 0.5 * (np.pi / 2) * _w

def PD_worm_N_weigth(N, sigma, V0):
    N_grid, D_L = Shulz_Zimm(N, sigma)
    #N_grid, D_L = Gauss_PD(N, sigma)
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
    
    # Ensure N_agg remains positive and >10
    N_min = max(10.0, N_mean - 3 * sigma_N)
    N_max = N_mean + 3 * sigma_N
    
    N_grid = np.linspace(N_min, N_max, nodes)
    weights = np.exp(-0.5 * ((N_grid - N_mean) / sigma_N)**2)
    weights /= np.sum(weights)
    
    return N_grid, weights

def ell_micelle_chains(q, R_core, eps_core, SLD_core, SLD_chain, N, sigma, L0, L_per, R_xs, sigma_R_xs, N_agg, R_g_chain, d,alpha_dec):
    V_core = (4/3)*np.pi*eps_core*(R_core**3)
    chain_amp = psi_chains(q, N, sigma, L0, L_per, R_xs, sigma_R_xs)
    
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

    R_decorr = (4*R_core**2)/(N_agg*R_xs)    ### np.exp(-q_2d*R_decorr)*
    
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


def ell_micelle_chains_unnorm(q, R_core, eps_core, SLD_core, SLD_chain, N, sigma, L0, L_per, R_xs, sigma_R_xs, N_agg, R_g_chain, d,alpha_dec):
    V_core = (4/3)*np.pi*eps_core*(R_core**3)
    chain_amp = psi_chains(q, N, sigma, L0, L_per, R_xs, sigma_R_xs)
    
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

    #np.exp(-q_2d*R_decorr)*
    phase_chain = np.where(qr_chain_pos < 1e-6, 1.0, np.exp(-alpha_dec*q_2d*R_decorr)*np.sin(qr_chain_pos) / qr_chain_pos)   #np.exp(-q_2d*R_decorr)*
    #phase_chain /= phase_chain[0]

    a_chain = psi_chains(q, N, sigma, L0, L_per, R_xs, sigma_R_xs)
    p_chain = a_chain**2
    
    # Excluded volume interactions (Section 3.1.43) mapped dynamically to N_agg and R_core
    #sigma_surf = (N_agg * R_g_chain**2) / (4 * (R_core + R_g_chain)**2)
    #v_int = 1.42 * (sigma_surf**1.04)
    
    #P_chain_q = p_chain / (1 + v_int * p_chain)
    #P_chain_0 = 1.0 / (1 + v_int)
    
    int_core = np.sum((amp_core**2) * np.sin(alpha_2d) * weights, axis=1)
    int_core_chain = np.sum(amp_core * phase_chain * np.sin(alpha_2d) * weights, axis=1)
    int_chain_chain = np.sum((phase_chain**2) * np.sin(alpha_2d) * weights, axis=1)
    
    # Return unnormalized form factor P(q) with interacting chains
    ff_unnorm = ((M_core**2)*int_core
                      + N_agg*(M_chains**2)*p_chain
                      + 2*N_agg*M_core*M_chains*a_chain*int_core_chain
                      + N_agg*(N_agg-1)*(M_chains**2)*int_chain_chain*p_chain 
                     )
                 
    return np.nan_to_num(ff_unnorm, nan=0.0)
    
def Brush_mic(params, q, intensity, eps=None):
    parvals = params.valuesdict()
    
    # New parameterization
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

    R_g_chain = get_Rg_excl_vol(N_brush*L0, L_per)

    V_PNIPAM = (N_PNIPAM*187.6)*1e-3
    
    SLD_core = SLD_PNIPAM*(1-phi_core) + SLD_water*phi_core
    SLD_chain = SLD_PEG

    N_grid, N_weights = get_N_agg_dist(N_agg_mean, sigma_N_agg, nodes=31)
    
    P_q_mean = np.zeros_like(q)
    R_core_mean = 0.0

    for n_agg, w in zip(N_grid, N_weights):
        V_core_n = (loops * n_agg * V_PNIPAM) / (1 - phi_core)
        R_core_n = ((3 * V_core_n) / (4 * np.pi * eps_core))**(1/3)
        R_core_mean += R_core_n * w
        
        P_q_n = ell_micelle_chains_unnorm(
            q, R_core_n, eps_core, SLD_core, SLD_chain, 
            N_brush, sigma_brush, L0, L_per, R_xs, sigma_R_xs, 
            n_agg, R_g_chain, d,alpha_dec
        )
        P_q_mean += P_q_n * w

    conc_mM = 1e3*conc/(PD_worm_N_weigth(N_brush, sigma_brush, M_sc)+2*N_PNIPAM*113+M_auxillary)
    N_density = (conc_mM*1e-6)*sc.constants.N_A / N_agg_mean
    
    R_eff = (R_core_mean*(2+eps_core))/3 + 2*R_g_chain
    V_eff = (4/3)*np.pi*(R_eff)**3
    eta = N_density*(V_eff*1e-21)
    
    try:
        stf = js.sf.PercusYevick(q, R_eff, eta=eta)
    except:
        stf = js.sf.PercusYevick(q, 12, eta=0.09)

    calc = js.dataArray(np.array([q, P_q_mean]))
    
    unsmeared = N_density * calc * stf + bckg
    smeared_calc = js.sas.smear(unsmeared=unsmeared, beamProfile=beam_profile).Y    
    diff = (intensity - smeared_calc) / eps
    return diff

    
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

    R_g_chain = get_Rg_excl_vol(N_brush*L0, L_per)

    V_PNIPAM = (N_PNIPAM*187.6)*1e-3
    
    SLD_core = SLD_PNIPAM*(1-phi_core) + SLD_water*phi_core
    SLD_chain = SLD_PEG

    N_grid, N_weights = get_N_agg_dist(N_agg_mean, sigma_N_agg, nodes=31)
    
    P_q_mean = np.zeros_like(q)
    R_core_mean = 0.0

    for n_agg, w in zip(N_grid, N_weights):
        V_core_n = (loops * n_agg * V_PNIPAM) / (1 - phi_core)
        R_core_n = ((3 * V_core_n) / (4 * np.pi * eps_core))**(1/3)
        R_core_mean += R_core_n * w
        
        P_q_n = ell_micelle_chains_unnorm(
            q, R_core_n, eps_core, SLD_core, SLD_chain, 
            N_brush, sigma_brush, L0, L_per, R_xs, sigma_R_xs, 
            n_agg, R_g_chain, d,alpha_dec
        )
        P_q_mean += P_q_n * w

    conc_mM = 1e3*conc/(PD_worm_N_weigth(N_brush, sigma_brush, M_sc)+2*N_PNIPAM*113+M_auxillary)
    N_density = (conc_mM*1e-6)*sc.constants.N_A / N_agg_mean
    
    R_eff = (R_core_mean*(2+eps_core))/3 + 2*R_g_chain
    V_eff = (4/3)*np.pi*(R_eff)**3
    eta = N_density*(V_eff*1e-21)
    
    try:
        stf = js.sf.PercusYevick(q, R_eff, eta=eta)
    except:
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



n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))


if __name__ == '__main__':

    D2O_2mm = D11_SANS('./IQ072126_c16.dat')
    NIPAM_596_24_0p6_40C = D11_SANS('./IQ072237_c38.dat')
    
    np.random.seed(465865)
    data = copy.deepcopy(NIPAM_596_24_0p6_40C)  # NIPAM_596_24_0p6_40C
    data.I =  NIPAM_596_24_0p6_40C.I-1*D2O_2mm.I  #0.89
    pars = Parameters()
    
    
    pars.add('N_agg', value = 20 , min=15, max=150, vary=True)
    pars.add('sigma_N_agg', value = 5 , min=1, max=25, vary=False)
    
    pars.add('phi_core', value = 0.50 , min=0, max=0.95, vary=True)
    #pars.add('R_core', value = 25,  min=2, max=50,vary=True)
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
    #pars.add('R_g_chain', value=28, min=24, max=32, vary=False)
    pars.add('sigma_R_xs', value = 0.5 , min=0.1, max=4, vary=False)
    
    
    pars.add('bckg', value =  0.00346, min=1e-7, max=0.1, vary=True)
    pars.add('conc', value = 0.625 , min=1e-10, max=16, vary=False)
    pars.add('SLD_brush', value = SLD_PEG , min=0.1, max=0.8, vary=False)
    
    
    low_ctf = 1
    high_ctf = 190
    
    fitted_pars = minimize(Brush_mic, pars, args=(data.q[low_ctf:high_ctf], data.I[low_ctf:high_ctf], data.I_err[low_ctf:high_ctf]), method='least_squares')  
    
    
    ax.loglog(data.q[low_ctf:high_ctf], plot_Brush_mic(fitted_pars, data.q[low_ctf:high_ctf]),'--',linewidth=2, zorder=2, label='Fit')
    ax.set_xlabel('$q, \\mathrm{nm}^{-1}$')
    ax.set_ylabel('$\\Delta\\Omega, \\mathrm{cm}^{-1}$')
    
    
    with open(f"fit_params_{data.sample_name}.dat", 'w') as f:
        f.write(str(fit_report(fitted_pars)))
    
    
    with Pool(n_workers) as pool:
        res = minimize(Brush_mic,args=(data.q[low_ctf:high_ctf], data.I[low_ctf:high_ctf], data.I_err[low_ctf:high_ctf]), method='emcee', nan_policy='omit', burn=200, steps=3200, thin=1, params=fitted_pars.params, is_weighted=True, progress=False, workers=pool)

    truths = res.params.valuesdict()
    truths.pop('SLD_brush')
    truths.pop('conc')
    truths.pop('loops')
    truths.pop('sigma')
    truths.pop('N_brush')
    #truths.pop('sigma_R_xs')
    #truths.pop('bckg')
    #truths.pop('L')
    emcee_plot = corner.corner(res.flatchain, labels=res.var_names,
                                truths=list(truths.values()))
    emcee_plot.savefig(f"{data.sample_name}_MCMC_out.png", dpi=300)