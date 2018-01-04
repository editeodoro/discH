from discH import galpotential
from discH.src.pot_disc.pot_disc import disc
from discH.src.pot_halo.pot_halo import halo
from discH.dynamic_component import isothermal_halo, Exponential_disc, Frat_disc, PolyExponential_disc, Gaussian_disc, NFW_halo, alfabeta_halo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import emcee
from  discH import discHeigth
import discH


d0=1e6
rc=3
mcut=100
alfa=1
beta=3
e=0
R=np.linspace(0,30,3)
Z=np.linspace(0,1,3)

n=NFW_halo(d0=d0,rs=rc,mcut=mcut,e=e)
g=n.potential(R,Z,grid=True)

print(g)

a=alfabeta_halo(d0=d0,rs=rc,mcut=mcut,e=e,alfa=alfa,beta=beta)
g=a.potential(R,Z,grid=True)

print(g)