from discH import galpotential
from discH.src.pot_disc.pot_disc import disc
from discH.src.pot_halo.pot_halo import halo
from scipy.integrate import quad
from discH.dynamic_component import einasto_halo, isothermal_halo, NFW_halo, hernquist_halo, deVacouler_like_halo, alfabeta_halo, plummer_halo ,Exponential_disc, Frat_disc, PolyExponential_disc, Gaussian_disc
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import emcee
from  discH import discHeight
import time

R=np.linspace(0.1,20,100)




'''
#Iso Halo
h_d0=152.8e6
h_rc=1.49
h_e=0
halo=isothermal_halo(d0=h_d0,rc=h_rc,e=h_e)
vc=halo.vcirc(R)
plt.plot(vc[:,0],vc[:,1],label='e=0')

h_e=0.2
halo=isothermal_halo(d0=h_d0,rc=h_rc,e=h_e)
vc=halo.vcirc(R)
plt.plot(vc[:,0],vc[:,1],label='e=0.2')

h_e=0.4
halo=isothermal_halo(d0=h_d0,rc=h_rc,e=h_e)
vc=halo.vcirc(R)
plt.plot(vc[:,0],vc[:,1],label='e=0.4')

h_e=0.6
halo=isothermal_halo(d0=h_d0,rc=h_rc,e=h_e)
vc=halo.vcirc(R)
plt.plot(vc[:,0],vc[:,1],label='e=0.6')

h_e=0.8
halo=isothermal_halo(d0=h_d0,rc=h_rc,e=h_e)
vc=halo.vcirc(R)
plt.plot(vc[:,0],vc[:,1],label='e=0.8')

plt.show()
'''

'''
#NFW Halo
h_c=9.9
h_V200=109.5
h_e=0
halo=NFW_halo.cosmo(c=h_c,V200=h_V200,e=h_e)
vc=halo.vcirc(R)
plt.plot(vc[:,0],vc[:,1],label='e=0')

h_e=0.2
halo=NFW_halo.cosmo(c=h_c,V200=h_V200,e=h_e)
vc=halo.vcirc(R)
plt.plot(vc[:,0],vc[:,1],label='e=%.1f'%h_e)

h_e=0.4
halo=NFW_halo.cosmo(c=h_c,V200=h_V200,e=h_e)
vc=halo.vcirc(R)
plt.plot(vc[:,0],vc[:,1],label='e=%.1f'%h_e)

h_e=0.6
halo=NFW_halo.cosmo(c=h_c,V200=h_V200,e=h_e)
vc=halo.vcirc(R)
plt.plot(vc[:,0],vc[:,1],label='e=%.1f'%h_e)

h_e=0.8
halo=NFW_halo.cosmo(c=h_c,V200=h_V200,e=h_e)
vc=halo.vcirc(R)
plt.plot(vc[:,0],vc[:,1],label='e=%.1f'%h_e)

plt.legend()
plt.show()
'''

'''
#Plummer halo
h_mass=4e10
h_rc=10
h_mcut=100
h_el=[0., 0.2,0.4,0.6,0.8,0.95]

for h_e in h_el:
    halo_pl=plummer_halo(mass=h_mass, rc=h_rc, e=h_e, mcut=h_mcut)
    vc=halo_pl.vcirc(R)
    #print(vc)
    plt.plot(vc[:,0],vc[:,1],label='e=%.1f'%h_e)

plt.legend()
plt.show()
'''

'''
#Pseudo halo
h_d0=4e7
h_alfa=0
h_beta=2
h_rc=0.5
h_mcut=100
h_el=[0.,0.2, 0.4, 0.6, 0.8, 0.95 ]
for h_e in h_el:
    halo_ab=alfabeta_halo(d0=h_d0,rs=h_rc, alfa= h_alfa, beta=h_beta, e=h_e, mcut=h_mcut)
    vc=halo_ab.vcirc(R)
    plt.plot(vc[:,0],vc[:,1],label='Pseudo-iso e=%.1f'%h_e)
    halo_iso=isothermal_halo(d0=h_d0,rc=h_rc, e=h_e, mcut=h_mcut)
    vc=halo_iso.vcirc(R)
    plt.plot(vc[:,0],vc[:,1],'--',label='iso e=%.1f'%h_e)

plt.legend()
plt.show()
'''
'''
#Hernquist
h_d0=4e8
h_rs=4
h_mcut=100
h_el=[0.,0.2, 0.4, 0.6, 0.8, 0.95 ]
for h_e in h_el:
    halo_he=hernquist_halo(d0=h_d0,rs=h_rs, e=h_e, mcut=h_mcut)
    vc=halo_he.vcirc(R)
    plt.plot(vc[:,0],vc[:,1],label='e=%.1f'%h_e)
plt.legend()
plt.show()
'''

'''
#DeVacouler
h_d0=4e8
h_rs=2
h_mcut=100
h_el=[0.,0.2, 0.4, 0.6, 0.8, 0.95 ]
for h_e in h_el:
    halo_he=deVacouler_like_halo(d0=h_d0,rs=h_rs, e=h_e, mcut=h_mcut)
    print(halo_he.alfa, halo_he.beta)
    vc=halo_he.vcirc(R)
    plt.plot(vc[:,0],vc[:,1],label='e=%.1f'%h_e)
plt.legend()
plt.show()
'''

'''
#Einasto
h_de=8.04e5
h_rs=18.85
h_el=[0.,0.2, 0.4, 0.6, 0.8, 0.95 ]
h_mcut=100
h_n=2.9
for h in h_el:
    halo=einasto_halo.de(de=h_de,rs=h_rs,n=h_n,e=h,mcut=h_mcut)
    vc=halo.vcirc(R)
    plt.plot(vc[:,0],vc[:,1],label='e=%.1f'%h)
plt.legend()
plt.show()
'''

#compare Einasto NFW
he_de=8.04e5
he_rs=18.85
he_n=2.9

hn_d0=1.80e7
hn_rs=10.74

e=0
mcut=100
toll=1e-4

halo = einasto_halo.de(de=he_de, rs=he_rs, n=he_n, e=e, mcut=mcut)
vc = halo.vcirc(R)
plt.plot(vc[:, 0], vc[:, 1], label='Einasto')


halo=NFW_halo(d0=hn_d0, rs=hn_rs, e=e, mcut=mcut)
vc=halo.vcirc(R)
plt.plot(vc[:,0],vc[:,1],label='NFW')
plt.legend()
plt.show()