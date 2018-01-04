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
import matplotlib as mpl

label_size =18
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
mpl.rcParams['mathtext.default']='regular'
mpl.rcParams.update({'figure.autolayout': True})
mpl.rcParams['contour.negative_linestyle'] = 'solid'
mpl.rcParams['axes.facecolor'] = 'white'

#from oldcpotlib import cpot_disk, potential_disk,  zexp, zgau, PolyExp, potentialh_iso, potentialh_nfw, cpot_halo, gasHeight
import discH
R=np.linspace(0.01,20,50)
fig=plt.figure()
ax=fig.add_subplot(111)
fig2=plt.figure()
ax2=fig2.add_subplot(111)
fig3=plt.figure()
ax3=fig3.add_subplot(111)

#Star
s_sigma0=2e6
s_Rd=2
s_zd=0.2
s_Rcut=30
s_zcut=10
star=Exponential_disc.thick(sigma0=s_sigma0,Rd=s_Rd,zlaw='sech2',zd=s_zd,Rcut=s_Rcut,zcut=s_zcut)

#Halo
h_d0=1e7
h_rc=7
h_e=0
mcut=80
toll=1e-4
halo=isothermal_halo(d0=h_d0,rc=h_rc,e=h_e,mcut=mcut)

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
v=gas.vcirc(R, nproc=3)
ax.plot(v[:,0],v[:,1],label='Thin ini')
v=gas.flare(R)
print(R)
ax2.plot(v[:,0],v[:,1],label='Thin ini')
v=gas.Sdens(R)
ax3.plot(v[:,0],v[:,1],label='Thin ini')



Z=np.logspace(np.log10(0.01),np.log10(25),40)-0.01
dh=discHeight(disc_component=gas, dynamic_components=(halo,halo2,star))
comp, tabzd, fzd, fitfunc=dh.height(flaw='poly',polyflare_degree=5,zlaw='gau',Rpoints=R,Zpoints=Z,inttoll=toll,mcut=None,Rcut=None,zcut=None,Rlimit='max')
print('tab',tabzd)
print(comp)
print(type(comp))
v=comp.vcirc(R, nproc=3)
ax.plot(v[:,0],v[:,1],label='Flare')
v=comp.flare(R)
ax2.plot(v[:,0],v[:,1],label='flare zd')
v=comp.flare(R,HWHM=True)
ax2.plot(v[:,0],v[:,1],label='flare HWHM')
ax2.scatter(tabzd[:,0],tabzd[:,1])

print('flare',v)
v=comp.Sdens(R)
ax3.plot(v[:,0],v[:,1],label='flare')

g_sigma0=1e6
g_Rd=5
g_alpha=2
g_zlaw='gau'
g_Rcut=50
g_zcut=50
gas=Frat_disc.thick(sigma0=g_sigma0,Rd=g_Rd,Rd2=g_Rd,alpha=g_alpha,zd=0.1,Rcut=g_Rcut,zcut=g_zcut)
v=gas.vcirc(R, nproc=3)
#ax.plot(v[:,0],v[:,1],label='Thin')


ax.legend()
ax.set_xlabel('R [kpc]',fontsize=20)
ax.set_ylabel('$V_c$ [km/s]', fontsize=20)
ax2.legend()
ax3.legend()
fig.savefig('a.pdf')
fig2.savefig('a2.pdf')
fig3.savefig('a3.pdf')
#tabf,flarelaw,flarelaw_der=gasHeight(R,Z,hlaw='iso',d0=h_d0,rc=h_rc,e=h_e,rcut=mcut/np.sqrt(2), g_sigma0=1e5,g_rcut=100.,g_zcut=10.,g_rlaw=lambda x: np.exp(-x/4), g_zlaw='gau', ext_potential=None ,disp=lambda x: 10*x/x, Nmax=10, ftoll=0.001, flarefit='pol',nproc=1,toll=1E-4,outdir='gasHeight')

