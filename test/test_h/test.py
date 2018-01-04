from discH import galpotential
from discH.src.pot_disc.pot_disc import disc
from discH.src.pot_halo.pot_halo import halo
from discH.dynamic_component import isothermal_halo, NFW_halo, plummer_halo, einasto_halo, Exponential_disc, Frat_disc, PolyExponential_disc, Gaussian_disc
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import emcee
from  discH import discHeight


#Star
s_sigma0=2e6
s_Rd=2
s_zd=0.2
s_Rcut=30
s_zcut=10
star=Exponential_disc.thick(sigma0=s_sigma0,Rd=s_Rd,zlaw='sech2',zd=s_zd,Rcut=s_Rcut,zcut=s_zcut)

#Halo
h_d0=1.80e7
h_de=8.04e5
h_re=18.85
h_rc=10.74
h_n=2.9
h_e=0
mcut=80
toll=1e-4
halo=einasto_halo.de(de=h_de,rs=h_re,n=h_n,e=h_e,mcut=mcut)
print(halo)
#halo=NFW_halo(d0=h_d0, rs=h_rc, e=h_e, mcut=mcut)

#Halo2
h2_d0=2e7
h2_rc=1.5
h2_e=0.9
h2_alfa=1
h2_beta=3
mcut=10
toll=1e-4
halo2=plummer_halo(d0=h2_d0,rc=h2_rc,e=h2_e,mcut=mcut)


g_sigma0=1e6
g_Rd=5
g_alpha=2
g_zlaw='gau'
g_Rcut=50
g_zcut=50
gas=Frat_disc.thin(sigma0=g_sigma0,Rd=g_Rd,Rd2=g_Rd,alpha=g_alpha,Rcut=g_Rcut,zcut=g_zcut)

R=np.linspace(0.01,20,10)
Z=np.logspace(np.log10(0.01),np.log10(25),40)-0.01
dh=discHeight(disc_component=gas, dynamic_components=(halo,halo2,star))
comp=dh.height(flaw='poly',polyflare_degree=5,zlaw='gau',Rpoints=R,Zpoints=Z,inttoll=toll,mcut=None,Rcut=None,zcut=None,Rlimit='max')[0]