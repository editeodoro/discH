import numpy as np
import matplotlib.pyplot as plt
from discH.dynamic_component import Exponential_disc
import time
from scipy.integrate import nquad, quad
from scipy.special import iv, kv, ellipe, ellipk
R=np.linspace(0.1,30,25)


def thin_disc(R,sigma0,Rd):

    G = 4.518359396265313e-39  # kpc^3/(msol s^2)
    kpc_to_km = 3.08567758e16  # kpc_to_km

    y=0.5*R/Rd


    cost=4*np.pi*Rd*sigma0*G
    a=iv(0,y)*kv(0,y) - iv(1,y)*kv(1,y)

    vv=np.sqrt(cost*y*y*a)*kpc_to_km
    print(vv)

    return vv

def drad(R,Rd):

    return np.exp(-R/Rd)

def dz(z,zd):

    return np.exp(-z/zd)/(2*zd)

def flare(R,z0,m):

    return z0+m*R


def drad_der(R, Rd):

    return -np.exp(-R/Rd)/Rd

def dz_der(z,zd):

    return -np.exp(-z/zd)/(2*zd*zd)

def flare_der(R,m):

    return m

def  dens(R, z, Rd, z0, m):

    zd=flare(R,z0,m)

    return drad(R,Rd)*dz(z,zd)

def dens_der(R, z, Rd, z0, m):

    zd=flare(R,z0,m)
    #print(zd)

    A=drad_der(R, Rd)*dz(z, zd)
    B=drad(R, Rd)*dz_der(z, zd)*flare_der(R,m)
    #print(B)

    #print(A,B)

    return A+B

def dens_der_thin(R, Rd):

    return drad_der(R, Rd)

def integrand(u,l,R,Rd, z0, m):

    if u==0 or R==0:
        return 0
    x=(u*u+R*R+l*l)/(2*R*u)
    if x==1:
        return 0
    p=x-np.sqrt(x*x-1)

    A=np.sqrt(u/p)
    B=ellipk(p*p) - ellipe(p*p)
    C=dens_der(u,l,Rd,z0,m)

    #print(A,B,C)

    return A*B*C

def integrand_thin(u,R,Rd):

    if u==0 or R==0:
        return 0
    x=(u*u+R*R)/(2*R*u)
    if x==1:
        return 0
    p=x-np.sqrt(x*x-1)

    A=np.sqrt(u/p)
    B=ellipk(p*p) - ellipe(p*p)
    C=dens_der_thin(u,Rd)

    #print(A,B,C)

    return A*B*C

def vcore(R,Rd,z0,m,Rcut,zcut,toll=1e-4):

    intc=nquad(integrand,((0,Rcut),(0,zcut)),args=(R,Rd,z0,m),opts=[({'points':[0,R],'epsabs':toll,'epsrel':toll}),({'points':[0,],'epsabs':toll,'epsrel':toll})])

    return intc[0]

def vcore_thin(R,Rd,Rcut,toll=1e-4):

    intc=quad(integrand_thin, 0, Rcut, args=(R,Rd), epsabs=toll, epsrel=toll,points=(0,R))

    return intc[0]/2.

def vcirc(R, sigma0, Rd, z0,m, Rcut, zcut, toll=1e-4):

    G=4.518359396265313e-39 #kpc^3/(msol s^2)
    kpc_to_km = 3.08567758e16  # kpc_to_km
    cost=-(8*G*sigma0)
    ret=np.zeros(shape=(len(R),2))
    ret[:,0]=R

    i=0
    for r in R:

        vc=vcore(r,Rd, z0, m, Rcut, zcut, toll)*cost*np.sqrt(r)
        ret[i,1]=vc
        print(i,r)
        i+=1


    ret[:,1]=np.sqrt(ret[:,1])*kpc_to_km

    return ret

def vcirc_thin(R, sigma0, Rd,  Rcut,  toll=1e-4):

    G=4.518359396265313e-39 #kpc^3/(msol s^2)
    kpc_to_km = 3.08567758e16  # kpc_to_km
    cost=-(8*G*sigma0)
    ret=np.zeros(shape=(len(R),2))
    ret[:,0]=R

    i=0
    for r in R:

        vc=vcore_thin(r,Rd, Rcut,  toll)*cost*np.sqrt(r)
        ret[i,1]=vc
        print(i,r)
        i+=1


    ret[:,1]=np.sqrt(ret[:,1])*kpc_to_km

    return ret

'''
Rd=2
z0=0.05
m=0.
sigma0=1e8
Rcut=50
zcut=20
v=vcirc(R,sigma0, Rd, z0, m, Rcut, zcut)
plt.plot(R, v[:,1],label='z0=%.3f, m=%.3f'%(z0,m))

Rd=2
m=0.1
sigma0=1e8
Rcut=50
zcut=20
v=vcirc(R,sigma0, Rd, z0, m, Rcut, zcut)
plt.plot(R, v[:,1],label='z0=%.3f, m=%.3f'%(z0,m))

star=Exponential_disc.thick(sigma0=sigma0,Rd=Rd,zlaw='exp',zd=z0,Rcut=Rcut,zcut=zcut)
t1=time.time()
v=star.vcirc(R, nproc=3)
t2=time.time()
plt.plot(v[:,0],v[:,1],'--',label='module constant',c='black')
print(t2-t1)


star=Exponential_disc.polyflare(sigma0=sigma0, Rd=Rd, Rlimit=4, polycoeff=(z0,m),zlaw='exp')
t1=time.time()
v=star.vcirc(R, nproc=3)
t2=time.time()
plt.plot(v[:,0],v[:,1],'--',label='module flare')
print(t2-t1)

plt.plot(R, thin_disc(R,sigma0,Rd), label='THin disc')





a=np.loadtxt('galforces_vc_h=0.dat')
plt.scatter(a[:,0],a[:,1],label='Gal zd=0')
a=np.loadtxt('galforces_vc_h=0.3.dat')
plt.scatter(a[:,0],a[:,1],label='Gal zd=0.3')
a=np.loadtxt('galforces_vc_h=0.6.dat')
plt.scatter(a[:,0],a[:,1],label='Gal zd=0.6')
a=np.loadtxt('galforces_vc_h=1.dat')
plt.scatter(a[:,0],a[:,1],label='Gal zd=1')
'''

Rd=2
z0=0.01
#z0=0.001
m=0.
sigma0=1e8
Rcut=50
zcut=20
v=vcirc(R,sigma0, Rd, z0, m, Rcut, zcut)
plt.plot(R, v[:,1],label='z0=%.3f, m=%.3f 2D'%(z0,m))

v=vcirc_thin(R, sigma0, Rd, Rcut)
plt.plot(R, v[:,1],label='3D')

plt.legend()
plt.show()




