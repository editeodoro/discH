from discH.src.pot_disc import  potential_disc, Exponential_disc
#from discH.src.pot_disc.pot_c_ext.rdens_law import poly_exponentialpy
#from discH.src.pot_disc.pot_c_ext.rflare_law import poly_flarepy
import numpy as np
from oldcpotlib import cpot_disk, potential_disk,  zexp, zgau, PolyExp, potentialh_iso, potentialh_nfw

import discH
import time
R=np.linspace(0.1,10,10)
Z=np.linspace(0.1,1,10)
Rd=3.5
sigma0=10
zd=0.2
dcoeff=np.zeros(10)
fcoeff=np.zeros(10)
dcoeff[0]=Rd
dcoeff[1]=1
fcoeff[0]=zd
fcoeff[1]=0
fcoeff[2]=0
#dcoeff=np.array([Rd,1,0,0,0,0,0])
#fcoeff=np.array([zd,1,2,0,0,0])
toll=1e-10


t1=time.time()
a=potential_disc(R,Z,sigma0=sigma0,rcoeff=dcoeff,fcoeff=fcoeff,zlaw='sech2', rlaw='epoly', flaw='poly', rcut=30,zcut=10,toll=toll,grid=True)
ttot=time.time()-t1
print(a,ttot)

t1=time.time()
ed=Exponential_disc.polyflare(sigma0=sigma0,Rd=Rd,polycoeff=(fcoeff[0],fcoeff[1],fcoeff[2]),zlaw='sech2')
a=ed.potential(R,Z,grid=True, toll=toll,Rcut=30, zcut=10,nproc=2)
ttot=time.time()-t1
print(a,ttot)

t1=time.time()
ed=Exponential_disc.asinhflare(sigma0=sigma0,Rd=Rd,h0=zd,c=1,Rf=5,zlaw='sech2')
a=ed.potential(R,Z,grid=True, toll=toll,Rcut=30, zcut=10,nproc=2)
ttot=time.time()-t1
print(a,ttot)

#t1=time.time()
#rdens=PolyExp(coef_list=[1,],rd=Rd)
#fdens=lambda x: fcoeff[0]+fcoeff[1]*x+fcoeff[2]*x*x
#a=cpot_disk(R, Z, sigma0=sigma0,zlaw='exp', rlaw=rdens.dens,flarelaw=fdens,rcut=30,zcut=10,toll=toll)
#a=potential_disk(R,Z,sigma0=sigma0,denslaw=zexp, rlaw=rdens.dens,flaw=fdens,rcut=30,zcut=10,toll=toll)
#ttot=time.time()-t1
#print(a,ttot)

'''
d0=10.
rc=4.
e=0.
mcut=100.

a = discH.isothermal_halo(d0, rc, e, mcut)
t1 = time.time()
b = a.potential(R, Z, grid=False, toll=toll)
ttot=time.time()-t1
print(b,ttot)



t1 = time.time()
b=potentialh_iso(R,Z,d0,rc,e,mcut/np.sqrt(2),toll)
ttot=time.time()-t1
print(b,ttot)

a = discH.NFW_halo(d0, rc, e, mcut)
t1 = time.time()
b = a.potential(R, Z, grid=False, toll=toll)
ttot=time.time()-t1
print(b,ttot)



t1 = time.time()
b=potentialh_nfw(R,Z,d0,rc,e,mcut/np.sqrt(2),toll)
ttot=time.time()-t1
print(b,ttot)



a=sigma0*poly_exponentialpy(R,*dcoeff)
print(a)
rdens=PolyExp(coef_list=[1,],rd=Rd)
print(sigma0*rdens.dens(R))


b=poly_flarepy(R,*fcoeff)
print(b)
fdens=lambda x: zd#+x+2*x*x
print(fdens(R))
'''

