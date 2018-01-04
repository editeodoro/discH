from discH import galpotential
from discH.src.pot_disc.pot_disc import disc
from discH.src.pot_halo.pot_halo import halo
from discH.dynamic_component import isothermal_halo, Exponential_disc, Frat_disc, PolyExponential_disc, Gaussian_disc
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import emcee
'''
#Star
s_sigma0=20
s_Rd=2
s_zd=0.2
star=Exponential_disc.thick(sigma0=s_sigma0,Rd=s_Rd,zlaw='sech2',zd=s_zd)

#Halo
h_d0=0.2
h_rc=7
h_e=0
halo=isothermal_halo(d0=h_d0,rc=h_rc,e=h_e)

#Gas
g_sigma0=10
g_Rd=5
g_alpha=2
g_h0=0.2
g_c=1
g_Rf=10
gas=Frat_disc.asinhflare(sigma0=g_sigma0,Rd=g_Rd,Rd2=g_Rd,alpha=g_alpha,h0=g_h0,Rf=g_Rf,c=g_c,zlaw='gau')

d=galpotential(dynamic_components=(star,halo,gas))
d.dynamic_components_info()


R=np.linspace(0.001,30,5)
#Z=np.logspace(np.log10(0.001),np.log10(10+0.001),30)-0.001
Z=np.linspace(0,10,5)
Rcut=30
zcut=30
mcut=100
toll=1e-4
g=d.potential(R,Z, toll=toll, grid=True, Rcut=Rcut, zcut=zcut, mcut=mcut, nproc=3)
d.save('test.txt')
print(g)

gas=gas.change_flaring(flaw='thin')
d.remove_components(idx=(2,))
d.add_components(components=(gas,))
g=d.potential(R,Z, toll=toll, grid=True, Rcut=Rcut, zcut=zcut, mcut=mcut, nproc=3)
d.save('test2.txt')
print(g)
'''

f=lambda R: 10*np.exp(-R/5)*(1+R/2)**1.5
zh=np.random.normal(0.8,0.05,size=20)
zh=np.abs(zh)

arr=np.zeros(shape=(20,2))
arr[:,0]=np.linspace(0.5,30,20)
arr[:,1]=f(arr[:,0])

arrz=np.zeros_like(arr)
arrz[:,0]=arr[:,0]
arrz[:,1]=zh
plt.scatter(arrz[:,0],zh)
Rplot=np.linspace(0,100,1000)

g=Gaussian_disc.thin(rfit_array=arr)
print(g)

g=g.change_flaring(flaw='thick',ffit_array=arrz)
print(g)
func_fit = lambda R,arr: np.where(R==0,arr[0],arr[0])
y=func_fit(Rplot,g.fparam)
plt.plot(Rplot,y,label='thick')

g=g.change_flaring(flaw='poly',ffit_array=arrz,fitdegree=4,Rlimit=30)
print(g)
func_fit = lambda R,arr: arr[0]+arr[1]*R+arr[2]*R*R+arr[3]*R*R*R+arr[4]*R*R*R*R+arr[5]*R*R*R*R*R+arr[6]*R*R*R*R*R*R+arr[7]*R*R*R*R*R*R*R
y=func_fit(Rplot,g.fparam)
plt.plot(Rplot,y,label='poly4')

g=g.change_flaring(flaw='poly',ffit_array=arrz,fitdegree=0,Rlimit=30)
print(g)
func_fit = lambda R,arr: arr[0]+arr[1]*R+arr[2]*R*R+arr[3]*R*R*R+arr[4]*R*R*R*R+arr[5]*R*R*R*R*R+arr[6]*R*R*R*R*R*R+arr[7]*R*R*R*R*R*R*R
y=func_fit(Rplot,g.fparam)
plt.plot(Rplot,y,label='poly0')

g=g.change_flaring(flaw='poly',ffit_array=arrz,fitdegree=2,Rlimit=30)
print(g)
func_fit = lambda R,arr: arr[0]+arr[1]*R+arr[2]*R*R+arr[3]*R*R*R+arr[4]*R*R*R*R+arr[5]*R*R*R*R*R+arr[6]*R*R*R*R*R*R+arr[7]*R*R*R*R*R*R*R
y=func_fit(Rplot,g.fparam)
plt.plot(Rplot,y,label='poly2')

g=g.change_flaring(flaw='poly',ffit_array=arrz,fitdegree=7,Rlimit=30)
print(g)
func_fit = lambda R,arr: arr[0]+arr[1]*R+arr[2]*R*R+arr[3]*R*R*R+arr[4]*R*R*R*R+arr[5]*R*R*R*R*R+arr[6]*R*R*R*R*R*R+arr[7]*R*R*R*R*R*R*R
y=func_fit(Rplot,g.fparam)
plt.plot(Rplot,y,label='poly7')

g=g.change_flaring(flaw='asinh',ffit_array=arrz,Rlimit=30)
print(g)
func_fit = lambda R, arr: arr[0]+arr[2]*np.arcsinh(R*R/(arr[1]*arr[1]))
y=func_fit(Rplot,g.fparam)
plt.plot(Rplot,y)

g=g.change_flaring(flaw='tanh',ffit_array=arrz,Rlimit=30)
print(g)
func_fit = lambda R, arr: arr[0]+arr[2]*np.tanh(R*R/(arr[1]*arr[1]))
y=func_fit(Rplot,g.fparam)
plt.plot(Rplot,y)

plt.ylim(0.5,1)
plt.legend()
plt.show()

plt.clf()
plt.scatter(arr[:,0],arr[:,1])
f=lambda R,arr: np.exp(-0.5*( (R-arr[1])*(R-arr[1]) )/(arr[0]*arr[0]))

plt.plot(Rplot,g.sigma0*f(Rplot,g.rparam))
plt.show()