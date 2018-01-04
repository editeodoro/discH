import numpy as np
import matplotlib.pyplot as plt
from discH.dynamic_component import Exponential_disc
import time
from scipy.special import iv, kv
R=np.linspace(0.1,30,50)


def thin_disc(R,sigma0,Rd):

    G = 4.518359396265313e-39  # kpc^3/(msol s^2)
    kpc_to_km = 3.08567758e16  # kpc_to_km

    y=0.5*R/Rd


    cost=4*np.pi*Rd*sigma0*G
    a=iv(0,y)*kv(0,y) - iv(1,y)*kv(1,y)

    vv=np.sqrt(cost*y*y*a)*kpc_to_km
    print(vv)

    return vv


s_sigma0=1e8
s_Rd=2
s_Rcut=50
s_zcut=20
s_zd=0.01
s_law='exp'
'''
#Star
s_sigma0=1e8
s_Rd=2
s_Rcut=50
s_zcut=20
s_zd=0.01
s_law='exp'
star=Exponential_disc.thick(sigma0=s_sigma0,Rd=s_Rd,zlaw=s_law,zd=s_zd,Rcut=s_Rcut,zcut=s_zcut)
t1=time.time()
v=star.vcirc(R,Rcut=s_Rcut, zcut=s_zcut, nproc=3)
t2=time.time()
plt.plot(v[:,0],v[:,1],label='zd=%.2f'%s_zd)
print(t2-t1)

s_zd=0.1
s_law='exp'
star=Exponential_disc.thick(sigma0=s_sigma0,Rd=s_Rd,zlaw=s_law,zd=s_zd,Rcut=s_Rcut,zcut=s_zcut)
t1=time.time()
v=star.vcirc(R,Rcut=s_Rcut, zcut=s_zcut, nproc=3)
t2=time.time()
plt.plot(v[:,0],v[:,1],label='zd=%.2f'%s_zd)
print(t2-t1)

s_zd=0.3
star=Exponential_disc.thick(sigma0=s_sigma0,Rd=s_Rd,zlaw=s_law,zd=s_zd,Rcut=s_Rcut,zcut=s_zcut)
t1=time.time()
v=star.vcirc(R,Rcut=s_Rcut, zcut=s_zcut, nproc=3)
t2=time.time()
plt.plot(v[:,0],v[:,1],label='zd=%.2f'%s_zd)
print(t2-t1)


s_zd=0.6
star=Exponential_disc.thick(sigma0=s_sigma0,Rd=s_Rd,zlaw=s_law,zd=s_zd,Rcut=s_Rcut,zcut=s_zcut)
t1=time.time()
v=star.vcirc(R,Rcut=s_Rcut, zcut=s_zcut, nproc=3)
t2=time.time()
plt.plot(v[:,0],v[:,1],label='zd=%.2f'%s_zd)
print(t2-t1)


s_zd=1.0
star=Exponential_disc.thick(sigma0=s_sigma0,Rd=s_Rd,zlaw=s_law,zd=s_zd,Rcut=s_Rcut,zcut=s_zcut)
t1=time.time()
v=star.vcirc(R,Rcut=s_Rcut, zcut=s_zcut, nproc=3)
t2=time.time()
plt.plot(v[:,0],v[:,1],label='zd=%.2f'%s_zd)
print(t2-t1)
'''

s_zd=0.01
star=Exponential_disc.polyflare(sigma0=s_sigma0, Rd=s_Rd, polycoeff=(s_zd,0.01),zlaw=s_law,Rcut=s_Rcut,zcut=s_zcut)
t1=time.time()
print(star)
v=star.vcirc(R,Rcut=s_Rcut, zcut=s_zcut, nproc=3, toll=1e-8)
t2=time.time()
plt.plot(v[:,0],v[:,1],label='flare')
print(t2-t1)

#thin
v=thin_disc(R,s_sigma0,s_Rd)
plt.plot(R,v,label='thin teo')

s_sigma0=1e8
s_Rd=2
s_Rcut=50
star=Exponential_disc.thin(sigma0=s_sigma0,Rd=s_Rd,Rcut=s_Rcut)
t1=time.time()
v=star.vcirc(R, nproc=3)
t2=time.time()
plt.plot(v[:,0],v[:,1],'--',label='thin num')
print(t2-t1)

a=np.loadtxt('galforces_vc_h=0.dat')
plt.scatter(a[:,0],a[:,1],label='Gal zd=0')
a=np.loadtxt('galforces_vc_h=0.3.dat')
plt.scatter(a[:,0],a[:,1],label='Gal zd=0.3')
a=np.loadtxt('galforces_vc_h=0.6.dat')
plt.scatter(a[:,0],a[:,1],label='Gal zd=0.6')
a=np.loadtxt('galforces_vc_h=1.dat')
plt.scatter(a[:,0],a[:,1],label='Gal zd=1')


plt.legend()
plt.show()
