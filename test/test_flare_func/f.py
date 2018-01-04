from discH import galpotential
from discH.src.pot_disc.pot_disc import disc
from discH.src.pot_halo.pot_halo import halo
from discH.dynamic_component import isothermal_halo, Exponential_disc, Frat_disc, PolyExponential_disc, Gaussian_disc, NFW_halo, alfabeta_halo, plummer_halo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import emcee
from  discH import discHeight
#from oldcpotlib import cpot_disk, potential_disk,  zexp, zgau, PolyExp, potentialh_iso, potentialh_nfw, cpot_halo, gasHeight
import discH

R=np.linspace(0.01,20,50)
g_sigma0=1e6
g_Rd=5
g_alpha=2
g_zlaw='gau'
g_Rcut=50
g_zcut=50
fparam=(4.3e-01, 1.0e-01, -1.4e-02, 1.3e-03, -5.7e-05, 9.1e-07, 0.0e+00, 0.0e+00, 2.0e+01, 1.5e+00)
gas=Frat_disc(sigma0=g_sigma0,Rd=g_Rd, fparam=fparam ,Rd2=g_Rd,alpha=g_alpha, Rcut=g_Rcut,zcut=g_zcut)

v=gas.flare(R)
print(gas)
print(v)