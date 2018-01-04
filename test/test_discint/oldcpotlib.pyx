# cython: profile=True
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.special as sf
import scipy.interpolate as sp
from scipy.integrate import quad, nquad
import functools
import multiprocessing as mp
import matplotlib.pyplot as plt
import fitlib as ft
import os
import sys
from scipy.optimize import curve_fit
import emcee
import datetime
import time

#V1.2.5

'''
Module for the calc of potential of halo or disk:
HALO:
two model are supplied- Isothermal -NFW
they can be flatten thanks to the parameter e
The potential are calculated following eq. 2.88b by BT 1987
DISK:
Calculate the general potential of a disk with Rlaw=(1+1/rd)^alfa exp(-r/rd)
The vertical law can be choosen between: -Gau -Exp -Sech2
The radial behavior of the vertical scale height can be choosen between:
-Tanh: h0 + c*Tanh((r/rf)^2) and -Asinh: h0+c*Asinh((r/rf)^2)
The potential are calculated following Cuddeford
#NB in future version it could be nice to introduce a general density law both in R and in Z (for example using inteporlated functions)
'''

###################HALO#############################
#m:spheroidal radius, variable of integration (m=sqrt(r^2+z^2/q^2))
#R,Z: cylindrical coordinates
#e: ellipticity linked with flattening q=sqrt(1-e*e) e=0/q=1-Spherical simmetry e=1-q=0 Maximum flattening (Razor thin disk)

#Auxiliary functions:
cpdef double xi(double m,double R,double Z, double e):
    """
    Auxiliary functions
    :param m: spheroidal radius (m=sqrt(r^2+z^2/q^2))
    :param R: cylindrical Radius
    :param Z: cylindrical height
    :param e: ellipticity (e-0 Spherical system, e=1 maximal flattening)
    :return: auxiliary functions for integrations
    """
    return  (R*R+Z*Z+e*e*m*m+np.sqrt((e*m)**4-2*e*e*m*m*(R*R-Z*Z)+(R*R+Z*Z)*(R*R+Z*Z)))/(2*m*m)
cpdef double psi_iso(double d0,double rc,double m):
    """
    Auxiliary functions linked to density law iso:
    d=d0/(1+m/rc^2)
    :param d0: Central density at (R,Z)=(0,0) [Msol/kpc^3]
    :param rc: Core radius [Kpc]
    :param m: spheroidal radius
    :return:
    """
    return d0*rc*rc*(np.log(1+((m*m)/(rc*rc))))
cpdef double psi_nfw(double d0,double rc,double m):
    """
    Auxiliary functions linked to density law nfw:
    d=d0/((m/rc)*(1+m/rc)^2)
    :param d0: Typical density at m/rc=0.465571... (nfw diverges in the center)
    :param rc: Core radius [Kpc]
    :param m: spheroidal radius
    :return:
    """
    cost=-2
    val=(1/(1+m/rc))-1
    return cost*d0*rc*rc*val

#Integrand functions:
cpdef double integrandhiso(double m,double mcut,double R,double Z,double e,double d0,double rc):
    """
    Potential integrand for isothermal halo: d=d0/(1+m/rc^2)
    :param m: Spheroidal radius and integration variable
    :param mcut: Spheroidal radius where for m>mcut d=0
    :param R: Cylindrical coordinate
    :param Z: Cylindrical coordinate
    :param e: halo ellipticity
    :param d0: Central density
    :param rc: Core radius
    :return: integrand function
    """
    if m==0: return 0 #Xi diverge to infinity when m tends to 0, but the integrand tends to 0
    else:
        if (m<=mcut): psi=psi_iso(d0,rc,m)
        else: psi=psi_iso(d0,rc,mcut)
        num=xi(m,R,Z,e)*(xi(m,R,Z,e)-e*e)*np.sqrt(xi(m,R,Z,e)-e*e)*m*psi
        den=((xi(m,R,Z,e)-e*e)*(xi(m,R,Z,e)-e*e)*R*R)+(xi(m,R,Z,e)*xi(m,R,Z,e)*Z*Z)
        return num/den

cpdef double integrandhnfw(double m,double mcut,double R,double Z,double e,double d0,double rc):
    """
    Potential integrand for isothermal halo: d=d0/((m/rc)*(1+m/rc)^2)
    :param m: Spheroidal radius and integration variable
    :param mcut: Spheroidal radius where for m>mcut d=0
    :param R: Cylindrical coordinate
    :param Z: Cylindrical coordinate
    :param e: halo ellipticity
    :param d0: Typical density at m/rc=0.465571... (nfw diverges in the center)
    :param rc:  Core radius [Kpc]
    :return: integrand function
    """
    if m==0: return 0 #Xi diverge to infinity when m tends to 0, but the integrand tends to 0
    if (m<=mcut): psi=psi_nfw(d0,rc,m)
    else: psi=psi_nfw(d0,rc,mcut)
    num=xi(m,R,Z,e)*(xi(m,R,Z,e)-e*e)*np.sqrt(xi(m,R,Z,e)-e*e)*m*psi
    den=((xi(m,R,Z,e)-e*e)*(xi(m,R,Z,e)-e*e)*R*R)+(xi(m,R,Z,e)*xi(m,R,Z,e)*Z*Z)
    return num/den

#Potential calc
cpdef double potentialh_iso(double R,double Z,double d0, double rc,double e,double rcut,double toll):
    """
    Calculate the potential of a isothermal (spheroid d=d0/(1+m/rc^2))
    in the point (R,Z). Use the formula 2.88b in BT 1987.
    The integration is performed with the function quad in scipy.quad.
    :param R: Radius there to calc the potential (kpc)
    :param Z: Hieght where to calc the potential (kpc)
    :param d0: Central density (Msol/kpc^3)
    :param rc: Core radius (kpc)
    :param e: halo ellipticity
    :param rcut: Cut radius for the halo (mcut=sqrt(2)rcut) (kpc)
    :param toll: Abs and Rel tollerance on the integration (see scipy.quad reference) [1E-4]
    :return: Potential at R,Z in unity of Kpc^2/Myr^2
    """
    G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
    cost=2*np.pi*G
    mcut=rcut*np.sqrt(2)
    m0=np.sqrt(R*R+(Z*Z)/(1-e*e)) #Value of m at R,Z
    fint=functools.partial(integrandhiso,e=e,R=R,Z=Z,d0=d0,rc=rc,mcut=mcut)
    intpot=quad(fint,0.,m0,epsabs=toll,epsrel=toll)[0]
    if (e<=0.0001): return -cost*(psi_iso(d0,rc,mcut)-intpot)
    else: return -cost*(np.sqrt(1-e*e)/e)*(psi_iso(d0,rc,mcut)*np.arcsin(e)-e*intpot)

cpdef double potentialh_nfw(double R, double Z, double d0, double rc, double e, double rcut, double toll):
    """
    Calculate the potential of a isothermal  d=d0/((m/rc)*(1+m/rc)^2)
    in the point (R,Z). Use the formula 2.88b in BT 1987.
    The integration is performed with the function quad in scipy.quad.
    :param R: Radius there to calc the potential (kpc)
    :param Z: Hieght where to calc the potential (kpc)
    :param d0: Typical density at m/rc=0.465571... (nfw diverges in the center) (Msol/kpc^3)
    :param rc: scale radius (kpc)
    :param e: halo ellipticity
    :param rcut: Cut radius for the halo (mcut=sqrt(2)rcut) (kpc)
    :param toll: Abs and Rel tollerance on the integration (see scipy.quad reference) [1E-4]
    :return: Potential at R,Z in unity of Kpc^2/Myr^2
    """
    G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
    cost=2*np.pi*G
    mcut=rcut*np.sqrt(2)
    m0=np.sqrt(R*R+(Z*Z)/(1-e*e)) #Value of m at R,Z
    fint=functools.partial(integrandhnfw,e=e,R=R,Z=Z,d0=d0,rc=rc,mcut=mcut)
    intpot=quad(fint,0.,m0,epsabs=toll,epsrel=toll)[0]
    if (e<=0.0001): return -cost*(psi_nfw(d0,rc,mcut)-intpot)
    else: return -cost*(np.sqrt(1-e*e)/e)*(psi_nfw(d0,rc,mcut)*np.arcsin(e)-e*intpot)
###############################################################

#####################DISK####################################

#Radial special law:
class FratLaw:
    """
    Surface density law as appear in Fraternali,?? Sigma/Sigma0=Exp[-r/rd](1+r/rd)^alpha
    """
    def __init__(self, alpha, rd):
        self.a=alpha
        self.rd=rd
        self.name='FratLaw'

    def dens(self,r):
        """
        Surface density at radius r normalized for sigma0
        """
        x=r/self.rd
        return ((1+x)**self.a)*np.exp(-x)

    def densder(self,r):
        """
        Derivative of Surface density at radius r normalized for sigma0
        """
        x=r/self.rd
        de=((1+x)**self.a)*np.exp(-x)
        num=((self.a-1)*self.rd-r)
        den=(1+x)*self.rd*self.rd
        return de*(num/den)

    def diskmass(self,low,up,s0=1):
        cost=s0*self.rd*self.rd*np.exp(1)*np.pi*2

        a=sf.gamma(1+self.a)*sf.gammaincc(1+self.a,(self.rd+up)/self.rd)-sf.gamma(2+self.a)*sf.gammaincc(2+self.a,(self.rd+up)/self.rd)

        b=sf.gamma(1+self.a)*sf.gammaincc(1+self.a,(self.rd+low)/self.rd)-sf.gamma(2+self.a)*sf.gammaincc(2+self.a,(self.rd+low)/self.rd)
        return cost*(a-b)

    def peak(self):
        rpeak=self.rd*(self.a-1)
        p=self.dens(rpeak)
        return rpeak, p




    def info(self):
        """
        Print the information about this function
        """
        infos='Functional form: %s \nParameters:\nalpha= %.4f Rd= %.4f\n' % (self.name,self.a,self.rd)
        return infos

class SplineDens:
    """
    Spline of the radial density law
    """
    def __init__(self,x,y,w=None,k=3,s=None,ext=3):
        """
        :param tab: 2D tab with radii and surface density
        :param w: Weigth
        :param k: Spline degree [3]
        :param s: smoothing factor (0 for interpolation)
        :param ext: Value of the external points 0-Extrapolate, 1-Zeros, 2-Error, 3-Constant value as at boundaries
        """
        self.k=k

        if s==None: self.s=len(x)
        else: self.s=s

        if ext==0: self.ext='Extrapolate'
        elif ext==1: self.ext='Zeros'
        elif ext==2: self.ext='Raise ValueError'
        elif ext==3: self.ext='BOundary value'
        else: raise ValueError('Wrong ext')

        self.fint=sp.UnivariateSpline(x,y,w=w,k=k,s=s,ext=ext)
        self.fintder=self.fint.derivative(1)
        self.name='Spline'

    def dens(self,r):
        return self.fint(r)

    def densder(self,r):
        return self.fintder(r)

    def diskmass(self,low,up,s0=1):
        int_func=lambda r: self.fint(r)*(2*np.pi*r)
        int=quad(int_func,low,up)[0]
        return s0*int

    def info(self):
        """
        Print the information about this function
        """
        infos='Functional form: %s \nParameters:\nOrder= %i Smoothing factor= %i Ext= %s\n' % (self.name,self.k,self.s,self.ext)
        return infos

class PolyExp:
    """
    Radial Density law of the type p(r)*exp(-r/rd) where p(r) is a polynomial
    The degree of polynomial depends on the iterable coef_list used to initialize the object.
    In the coef_list the largest degree is the first the last is the constant value.
    E.g.: [3,4,1,2] stay for 3*x^3 + 4*x^2 + x + 2.
    """
    def __init__(self,coef_list,rd):
        self.coef=coef_list
        self.rd=rd
        self.polifunc=np.poly1d(coef_list)
        self.order=len(coef_list)-1
        self.name='PolyExp'

    def dens(self,r):
        x=r/self.rd
        return self.polifunc(r)*np.exp(-x)

    def densder(self,r):
        x=r/self.rd
        polider=np.polyder(self.polifunc)
        return (polider(r)- (self.polifunc(r)/self.rd) )*np.exp(-x)

    def diskmass(self,low,up,s0=1):
        int_func=lambda r: self.dens(r)*(2*np.pi*r)
        int=quad(int_func,low,up)[0]
        return s0*int

    def info(self):
        """
        Print the information about this function
        """
        infos='Functional form: %s \nParameters:\nRd= %.4f Pol order= %i coeff= ' % (self.name,self.rd,self.order)
        i=0
        for x in self.coef[::-1]:
            infos+='x%i: %.4f ' % (i,x)
            i+=1
        infos+='\n'
        return infos

#Radial special law:
class FratLawExp:
    """
    Surface density law as appear in Fraternali,?? Sigma/Sigma0=Exp[-r/rd](1+r/rd)^alpha
    """
    def __init__(self, s1, rd1,rd2,alpha):
        self.a=alpha
        self.rd1=rd1
        self.rd2=rd2
        self.s1=s1
        self.name='FratLawExp'

    def dens(self,r):
        """
        Surface density at radius r normalized for sigma0
        """
        x1=r/self.rd1
        x2=r/self.rd2
        return self.s1*((1+x1)**self.a)*np.exp(-x1)+(1-self.s1)*np.exp(-x2)

    def densder(self,r):
        """
        Derivative of Surface density at radius r normalized for sigma0
        """
        x1=r/self.rd1
        x2=r/self.rd2
        de=((1+x1)**self.a)*np.exp(-x1)
        num=((self.a-1)*self.rd1-r)
        den=(1+x1)*self.rd1*self.rd1
        a1=de*(num/den)*self.s1
        a2=np.exp(-x2)*(self.s1-1)/self.rd2
        return a1+a2

    def diskmass(self,low,up,s0=1):
        cost=s0*2*np.pi
        cost2=self.rd1*self.rd1*np.exp(1)*self.s1
        a=(self.rd2*np.exp(-(low+up)/2)*(self.s1-1))*(self.rd2*(np.exp(low/self.rd2) -  np.exp(up/self.rd2)) -np.exp(up/self.rd2)*low + np.exp(low/self.rd2)*up )
        b=-sf.gamma(1+self.a)*sf.gammaincc(1+self.a,(self.rd1+low)/self.rd1)-sf.gamma(2+self.a)*sf.gammaincc(2+self.a,(self.rd1+up)/self.rd1)
        c=sf.gamma(1+self.a)*sf.gammaincc(1+self.a,(self.rd1+up)/self.rd1)+sf.gamma(2+self.a)*sf.gammaincc(2+self.a,(self.rd1+low)/self.rd1)
        return cost*(a+cost2*(b+c))


    def info(self):
        """
        Print the information about this function
        """
        infos='Functional form: %s \nParameters:\nalpha= %.4f Rd= %.4f\n' % (self.name,self.a,self.rd)
        #return infos


#Vertical law only (Class)
#Gau
class Gau:
    """
    Gaussian vertical law
    """

    def dens(self,z,zd):
        """
        Normalized density (integral equal t 1) at height z with scale height zd
        """
        norm=1/(np.sqrt(2*np.pi)*zd)
        de=np.exp(-0.5*(z/zd)*(z/zd))
        return norm*de

    def densder(self,z,zd):
        """
        Derivative Normalized density (integral equal t 1) at height z with scale height zd
        """
        de=self.dens(z,zd)
        der=(z*z-zd*zd)/(zd*zd*zd)
        return de*der

class Sech2:
    """
    Hyperbolic secant vertical law
    """

    def dens(self,z,zd):
        """
        Normalized density (integral equal t 1) wrt to zd at height z with scale height zd
        """
        norm=(1/(2*zd))
        de=(1/(np.cosh(z/zd)) ) *  (1/(np.cosh(z/zd)) )
        return norm*de

    def densder(self,z,zd):
        """
        Derivative Normalized density (integral equal t 1)  wrt zd at height z with scale height zd
        """
        de=self.dens(z,zd)
        #der=(zd-z*np.tanh(z/zd)*np.tanh(z/zd))/(zd*zd)
        der=(zd-2*z*np.tanh(z/zd))/(zd*zd)
        return de*der





#Vertical Density law:
def zexp(u,l,rlaw,flaw):
    """
    Vertical law: Exp(-l/zd)/(2zd)
    l and zd need to have the same physical units
    :param u: integrand variable in R
    :param l: integrand variable in Z
    :param rlaw: function of one variable describing the radial surface law
    :param flaw: function of one variable describing the radial bheaviour of the flaring
    :return:
    """
    zd=flaw(u)
    norm=(1/(2*zd))
    densr=rlaw(u)
    densz=np.exp(-np.abs(l/zd))
    return densr*densz*norm

def zsech(u,l,rlaw,flaw):
    """
    Vertical law: Sech(-l/zd)^2
    l and zd need to have the same physical units
    :param u: integrand variable in R
    :param l: integrand variable in Z
    :param rlaw: function of one variable describing the radial surface law
    :param flaw: function of one variable describing the radial bheaviour of the flaring
    :return:
    """
    zd=flaw(u)
    norm=(1/(2*zd))
    densr=rlaw(u)
    densz=(1/(np.cosh(l/zd)) ) *  (1/(np.cosh(l/zd)) )
    return densr*densz*norm

def zgau(u,l,rlaw,flaw):
    """
    Vertical law: Exp(-0.5*(l/zd)^2)/(sqrt(2 pi) zd)
    l and zd need to have the same physical units
    :param u: integrand variable in R
    :param l: integrand variable in Z
    :param rlaw: function of one variable describing the radial surface law
    :param flaw: function of one variable describing the radial bheaviour of the flaring
    :return:
    """
    zd=flaw(u)
    norm=(1/(np.sqrt(2*np.pi)*zd))
    densr=rlaw(u)
    densz=np.exp(-0.5*(l/zd)*(l/zd))
    return densr*densz*norm


#Potential calc
def integrand(u,l,R,Z,denslaw,rlaw,flaw):
    """
    Integrand function
    :param u: integrand variable in R
    :param l: integrand variable in Z
    :param R: Radius where to calc the potential
    :param Z: Height where to calc the potential
    :param denslaw: One of the density law previuos described (zgau,zexp,zsech)
    :param rlaw: function of one variable describing the radial surface law
    :param flaw: function of one variable describing the radial bheaviour of the flaring
    :return:
    """
    if (R==0) | (u==0): return 0 #Singularity of the integral
    else:
        x=(R*R + u*u + (l-Z)*(l-Z))/(2*R*u)
        y=2/(x+1)
        if x==1: return 0
        else: return np.sqrt(u*y)*sf.ellipk(y)*denslaw(u,l,rlaw,flaw)

def potential_disk(R,Z,sigma0,denslaw,rlaw,flaw,rcut=30,zcut=10,toll=1E-4):
    """
    :param R: Radial where to calc the potential (kpc)
    :param Z: Height where to calc che potential (kpc)
    :param sigma0: Value of the intrinsic surface density at R=0 (Msol/kpc^2)
    :param rlaw: function of one variable describing the radial surface law__ NB it need to be normalized for the value at R=0
    :param flaw: function of one variable describing the radial bheaviour of the flaring
    :param denslaw: One of the previuos density law (zgau,zexp,zsech)
    :param rcut: Cut radius for integration along R
    :param zcut: Cut radius for integration along Z
    :param toll: Absolute and relative tollerance on the integration (see quad.scipy) [1E-4]
    :return: potential at point of cylindrical coordinates (R,Z)
    """
    G=4.498658966346282e-12 #kpc^2/(msol myr^2)
    cost=-(2*G*sigma0)/(np.sqrt(R))
    fint=functools.partial(integrand,R=R,Z=Z,denslaw=denslaw,rlaw=rlaw,flaw=flaw)
    value=nquad(fint,[[0.,rcut],[-zcut,zcut]],opts=[({'points':[0,R],'epsabs':toll,'epsrel':toll}),({'points':[Z],'epsabs':toll,'epsrel':toll})])
    #value=nquad(fint,[[0.0000,intlimit*rd],[-intlimit*h0,intlimit*h0]],opts=[({'points':[0,R]}),({'points':[Z]})])
    return cost*value[0]

###CALC of POTNTIAL GIVEN ARRAYS OF R AND Z
#disk
def cpot_disk(double[:] rtab,double[:] ztab, double sigma0, zlaw, rlaw, flarelaw,rcut,zcut,toll=1E-4):
    """
    Calculate the potential of a disk for a series of cylindrical coordinate (R,Z)
    :param rtab: numpy array with R coordinate
    :param ztab: numpy array  with Z coordinate
    :param sigma0: Central surface density (Msol/kpc^2)
    :param zlaw: z-density law (can be exp, sech, gau)
    :param rlaw: r-density law  (1 variable function)
    :param flarelaw: r-flare law (1 variable function)
    :param rcut: R where to stop the radial integration
    :param zcut: Z where to stop the vertical integration
    :param toll: Abs and rel tollerance in integration (see scipy.nquad)
    :return: A numpy array with three column and len(R)*len(Z) rows:
            col 0- R col 1- Z col 2- pot(R,Z)
    """
    cdef:
        int i,j, count, nr, nz, countr
        double [:,:] table


    #define integrand
    if (zlaw=='exp'): dlaw=zexp
    elif (zlaw=='gau'): dlaw=zgau
    elif (zlaw=='sech'): dlaw=zsech
    else: raise ValueError('wrong dlaw')


    table=np.zeros(shape=(len(rtab)*len(ztab),3),dtype=np.float64,order='C')

    nr=rtab.shape[0]
    nz=ztab.shape[0]
    count=0
    countr=0
    for i in range(nr):
        for j in range(nz):
            r=rtab[i]
            z=ztab[j]
            print('Now processing: %.3f %.3f' % (r,z))
            table[count,0]=r
            table[count,1]=z
            table[count,2]=potential_disk(r,z,sigma0,dlaw,rlaw,flarelaw,rcut,zcut,toll)
            count+=1
        countr=count
    return np.asarray(table)


#halo
def cpot_halo(double[:] rtab,double[:] ztab, double d0,double rc,double e, double rcut, hlaw ,toll=1E-4):
    """
    Calculate the potential of a halo for a series of cylindrical coordinate (R,Z)
    :param rtab: numpy array with R coordinate
    :param ztab: numpy array  with Z coordinate
    :param d0: Central density (Msol/kpc^3)
    :param rc: Core radius
    :param e: ellipticity
    :param rcut: Cut radius
    :param hlaw: Density law port the halo (iso or nfw)
    :param toll: Abs and rel tollerance in integration (see scipy.nquad)
    :return: A numpy array with three column and len(R)*len(Z) rows:
            col 0- R col 1- Z col 2- pot(R,Z)
    """

    cdef:
        int i,j, count, nr, nz, countr
        double [:,:] table

    #define integrand
    if (hlaw=='iso'): pot=potentialh_iso
    elif (hlaw=='nfw'): pot=potentialh_nfw
    else: raise ValueError('wrong dlaw')

    table=np.zeros(shape=(len(rtab)*len(ztab),3),dtype=np.float64,order='C')

    nr=rtab.shape[0]
    nz=ztab.shape[0]
    count=0
    countr=0
    for i in range(nr):
        for j in range(nz):
            r=rtab[i]
            z=ztab[j]
            print('Now processing: %.3f %.3f' % (r,z))
            table[count,0]=r
            table[count,1]=z
            table[count,2]=pot(r,z,d0,rc,e,rcut,toll)
            count+=1
        countr=count
    return np.asarray(table)

#####Calculate dens
def cdens(table,disp):
    """
    Calculate the  normalized vertical density given a table with R-Z-Pot.
    The d/d0 is calculated from the hydrostatical equlibrium as Exp(-(pot(R,Z)-pot(R,Zmin)/disp^2(R))
    NB. the table need to be sorted both in R and Z
    NB2. the reference value at pot(R,Zmin) is the value of the potential for the minimum Z found in the file
    :param table: 3 column numpy array
    :param disp: disp function of radius  (km/s)
    :return: a new numpy array with 4 column, the first three are the same of the input table, the last
            represents the normalized density d/d0 at coordinates R,Z.
    """
    conv= 1.02269012107e-3 #km/s to kpc/Myr
    newtable=np.zeros(shape=(len(table),4),dtype=np.float64,order='C')
    newtable[:,0:3]=table
    count=0
    i=0
    new=0
    while new!=table[-1,0]:
        old=table[i,0]
        new=table[i+1,0]
        if (new==old): count+=1
        else:
            r=newtable[i-count,0]
            pot0=newtable[i-count,2]
            newtable[i-count:i+1,3]=np.exp(-(newtable[i-count:i+1,2]-pot0)/(disp(r)*disp(r)*conv*conv))
            count=0
        i+=1
    r=newtable[i,0]
    pot0=newtable[i,2]
    newtable[i:,3]=np.exp(-(newtable[i:,2]-pot0)/(disp(r)*disp(r)*conv*conv))


    return newtable

################Rotation Curve###########################

#HALO
cpdef sph_to_ell(hlaw,d0,rc,e):
    """
    Find the value of d0 and rc for an halo of ellipticity e, starting from the values d0 and rc for a spherical halo.
    The transformation is performed to have the same rotation curve. (App. C, my thesis)
    :param hlaw: 'iso' for isothermal, 'nfw' for navarro-frank-white
    :param d0:
    :param rc:
    :param e:
    :return: d0(e), rc(e)
    """
    q=np.sqrt(1-e*e)
    print(q)
    if hlaw=='iso':
        kappa=0.598+(0.996/(q+1.460))-(0.003*q*q*q)
        lamb=0.538+(0.380/q)+0.083*q
    elif hlaw=='nfw':
        kappa=0.549+(1.170/(q+1.367))-0.047*q
        lamb=0.510+(0.312/q)+0.178*q
    else: raise IOError('hlaw allowed: iso or nfw')
    d0n=d0*lamb
    rcn=rc*kappa
    return d0n,rcn

cpdef nfw_to_plaw(c,v200,H=67):
    """
    To pass from the classical functional forms of the NFW density law with c and v200
    to the power law formulation with d0 and rc.
    :param c: Concentration parameter [kpc]
    :param v200: Rot velocity at R200 [km/s]
    :param H: Hubble parameter (km/s/Mpc)
    :return: d0 [Msol/kpc^3], rc [kpc]
    """
    num=14.93*(v200/100.)
    den=(c/10)*(H/67)
    rc=num/den
    fc=(c*c*c)/(np.log(1+c)-(c/(1+c)))
    d0=8340*(H/67.)*(H/67.)*fc
    return d0, rc

def rot_nfw(r,c,v200,H=67):
    """
    Rotation curve for a spherical nfw halo
    :param r: radii array (kpc)
    :param c:  Concentration factor
    :param v200: Velocity at R200 (km/s)
    :param H: Hubble param [67] (km/s/Mpc)
    :return: an array with the vrot in km/s
    """
    rs=(14.93*(v200/100))/((c/10)*(H/67))
    x=r/(rs)
    num=np.log(1+x)-(x)/(1+x)
    den=np.log(1+c) -(c)/(1+c)
    return v200*np.sqrt((c/x)*(num/den))

def rot_plaw(r,d0,rs):
    """
    Rotation curve for a spherical nfw halo (d=d0/(r/rs*(1+r/rs)^2))
    :param r: radii array (kpc)
    :param d0:  Typical density (Msol/kpc^3)
    :param rs: Scale radius (kpc)
    :return: an array with the vrot in km/s
    """
    G=4.302113488372941e-06 #kpc km2/(msol s^2)
    x=r/rs
    cost=4*np.pi*G*d0*rs*rs
    return np.sqrt((cost/x)*(np.log(1+x) - (x)/(1+x)))

def rot_iso(r,d0,rc):
    """
    Rotation curve for a spherical halo (d=d0/(1+r/rc)^2)
    :param r: radii array (kpc)
    :param d0: Central density (Msol/kpc^3)
    :param rc: Core radius (kpc)
    :return: an array with the vrot in km/s
    """
    G=4.302113488372941e-06 #kpc km2/(msol s^2)
    x=r/rc
    vinf2=4*np.pi*G*d0*rc*rc
    return np.sqrt(vinf2*(1-np.arctan(x)/x))

def d0_estimate(rc,vinf):
    G=4.302113488372941e-06 #kpc km2/(msol s^2)
    return (vinf*vinf)/(4*np.pi*G*rc*rc)


def halo_mass(mup,hlaw,d0,rc,e):
    q=np.sqrt(1-e*e)
    cost=4*np.pi*d0*q
    if hlaw=='iso':
        dens=lambda m:  (m*m)/(1+(m*m)/(rc*rc))
        minf=0.
    elif hlaw=='nfw':
        dens=lambda m:  (m*m)/(  (m/rc)*((1+(m)/(rc))*(1+(m)/(rc)) ) )
        minf=1e-6
    else: raise ValueError()
    int=quad(dens,minf,mup)[0]
    return cost*int



#TESTATI FUNZIONANO
#####################################################

###################################################
##################Disk#############################

#Analytical razor thin disk
cpdef double rot_razor_thin(r,sigma0,rd):
    """
    Analytical Rotation curve for a exponential razor thin disk
    :param r: Radius (kpc)
    :param sigma0: Central surface density (kpc)
    :param rd: Radial scale length (kpc)
    :return: Velocity in km/s
    """
    G=4.518359396265313e-39 #kpc^3/(msol s^2)
    x=r/(2*rd)
    cost=4*np.pi*G*sigma0*rd*x*x
    ker=sf.iv(0,x)*sf.kv(0,x)-sf.iv(1,x)*sf.kv(1,x)
    return np.sqrt(cost*ker)*3.08567758e16

#Rotcur for stars
#Exponential disk with gaussina vertical law with constant scale height
cpdef double integrand_disk_gau(u,  l,  r,  rd,  zd):
    """
    Integrand for rotcur with exponential radial law and gaussian vertical law
    :param u: radial integrand
    :param l: vertical integrand
    :param r: radius
    :param rd: radial scale length (kpc)
    :param zd: vertical scale length (kpc)
    :return:
    """
    if (r==0) | (u==0): return 0 #Singularity of the integral
    else:
        x=(r*r+u*u+l*l)/(2*r*u)
        p=x-np.sqrt(x*x-1)
        dens=-((np.exp(-u/rd))/rd)* ( (np.exp(-0.5*((l*l)/(zd*zd))))/(zd*np.sqrt(2*np.pi))  )
        ker=np.sqrt(u/p)*(sf.ellipk(p*p)-sf.ellipe(p*p))
        return dens*ker

cpdef double integrand_disk_sech2(u,  l,  r,  rd,  zd):
    """
    Integrand for rotcur with exponential radial law and sech2 vertical law
    :param u: radial integrand
    :param l: vertical integrand
    :param r: radius
    :param rd: radial scale length (kpc)
    :param zd: vertical scale length (kpc)
    :return:
    """
    if (r==0) | (u==0): return 0 #Singularity of the integrad
    else:
        x=(r*r+u*u+l*l)/(2*r*u)
        p=x-np.sqrt(x*x-1)
        dens=-((np.exp(-u/rd))/rd)* (((1/(np.cosh(l/zd)) ) *  (1/(np.cosh(l/zd)) ) ) / (2*zd) )
        ker=np.sqrt(u/p)*(sf.ellipk(p*p)-sf.ellipe(p*p))
        return dens*ker

cpdef double rotcur_sdisk_gau(r,sigma0,rd,zd,rcut,zcut,toll=1E-4):
    """
    Rotation curve in the plane at Z=0 for an exponential surface density and a gaussina vertical density
    :param r: Radius (kpc)
    :param sigma0: Central surface density (Msun/kpc^2)
    :param rd: Radial scale length (kpc)
    :param zd:  Vertical scale length (sigma of the gaussian) (kpc)
    :param rcut: Cut radius for the integration (kpc)
    :param zcut: Cut higth for the integration (kpc)
    :param toll: Abs and rell tollerance for the integration (see scipy.nquad)
    :return: Rotation curve in km/s
    """
    G=4.518359396265313e-39 #kpc^3/(msol s^2)
    cost=(sigma0*8*G)/(np.sqrt(r))
    fint=functools.partial(integrand_disk_gau,r=r,rd=rd,zd=zd)
    value=nquad(fint,[[0.00,rcut],[0.,zcut]],opts=[({'points':[0,r],'epsabs':toll,'epsrel':toll}),({'points':[0],'epsabs':toll,'epsrel':toll})])
    return np.sqrt(-r*cost*value[0])*3.08567758e16

cpdef double rotcur_sdisk_sech2(r,sigma0,rd,zd,rcut,zcut,toll=1E-4):
    """
    Rotation curve in the plane at Z=0 for an exponential surface density and a sech2 vertical density
    :param r: Radius (kpc)
    :param sigma0: Central surface density (Msun/kpc^2)
    :param rd: Radial scale length (kpc)
    :param zd:  Vertical scale length (zd of the sech2) (kpc)
    :param rcut: Cut radius for the integration (kpc)
    :param zcut: Cut higth for the integration (kpc)
    :param toll: Abs and rell tollerance for the integration (see scipy.nquad)
    :return: Rotation curve in km/s
    """
    G=4.518359396265313e-39 #kpc^3/(msol s^2)
    cost=(sigma0*8*G)/(np.sqrt(r))
    fint=functools.partial(integrand_disk_sech2,r=r,rd=rd,zd=zd)
    value=nquad(fint,[[0.00,rcut],[0.,zcut]],opts=[({'points':[0,r],'epsabs':toll,'epsrel':toll}),({'points':[0],'epsabs':toll,'epsrel':toll})])
    return np.sqrt(-r*cost*value[0])*3.08567758e16

#General Rotcur
cpdef double rotcur_kernel(u,l,r):
    """
    Kernel for the general rotcur
    :param u: radial integrand
    :param l: vertical integrand
    :param r:  Radius
    :return: kernel for the integration
    """
    if (r==0) | (u==0): return 0 #Singularity of the integral
    else:
        x=(r*r+u*u+l*l)/(2*r*u)
        p=x-np.sqrt(x*x-1)
        ker=np.sqrt(u/p)*(sf.ellipk(p*p)-sf.ellipe(p*p))
        return ker


def rotcur_disk_flare(r,sigma0,rlaw,rlawder,zlaw,zlawder,flaw,flawder,rcut,zcut,toll=1E-4):
    """
    :param r: Radius (kpc)
    :param sigma0: Central surface density (Msun/kpc^2)
    :param rlaw: Radial density law, it need to depend only on the radius, normalized to sigma0
    :param rlawder: Radial  Derivative of Radial density law, it need to depend only on the radius, normalized to sigma0
    :param zlaw: Vertical density law, it need to depend on the Z and Zd
    :param zlawder: Derivative of Vertical density law wrt Zd, it need to depend on the Z and Zd
    :param flaw: Radial behavior of Zd, it need to depend only on the radius
    :param flawder: Radial derivative of Zd, it need to depend only on the radius
    :param rcut: Cut radius for the integration (kpc)
    :param zcut: Cut higth for the integration (kpc)
    :param toll: Abs and rell tollerance for the integration (see scipy.nquad) [1E-4]
    :return: Velocity in km/s, If negative it means that the Radial force points toward the outskirt
    """
    G=4.518359396265313e-39 #kpc^3/(msol s^2)
    cost=(sigma0*8*G)/(np.sqrt(r))
    #INtegrand
    densderu= lambda u,l: rlawder(u)*zlaw(l,flaw(u)) + zlawder(l,flaw(u))*rlaw(u)*flawder(u)
    integrand= lambda u,l: densderu(u,l)*rotcur_kernel(u,l,r)
    value=nquad(integrand,[[0.00,rcut],[0.,zcut]],opts=[({'points':[0,r],'epsabs':toll,'epsrel':toll}),({'points':[0],'epsabs':toll,'epsrel':toll})],)
    if value[0]<0: return np.sqrt(-r*cost*value[0])*3.08567758e16
    else: return -np.sqrt(r*cost*value[0])*3.08567758e16


#prove
class ParDo:
    '''
    Manage Multiprocess things
    '''
    def __init__(self, nproc):
        '''
        nproc=number of processor involved
        '''
        self.n=nproc
        self.process=list(np.zeros(nproc))
        self.output=mp.Queue()

    def initialize(self):
        self.output=mp.Queue()

    def run(self,array,target):
        #Initialize process
        if self.n==1: self.process[0]=mp.Process(target=target, args=(array[:],))
        else:
            dim=int(len(array)/self.n)
            for i in range(self.n-1):
                start=int(dim*i)
                end=int(dim*(i+1))
                self.process[i]=mp.Process(target=target, args=(array[start:end],))
            self.process[-1]= mp.Process(target=target, args=(array[end:],))
        #Run
        for p in self.process:
            p.start()
        for p in self.process:
            p.join()
        results=np.concatenate([self.output.get() for p in self.process])
        indxsort=np.argsort(results[:,0], kind='mergesort')
        return results[indxsort]


def gasHeigth(R,Z,hlaw='iso',d0=None,rc=1.,e=0.,rcut=100,s_sigma0=None,s_rd=1.,s_zd=1.,s_rcut=100.,s_zcut=100.,s_zlaw='sech',g_sigma0=None,g_rcut=100.,g_zcut=10.,g_rlaw=lambda x: 0, g_zlaw='gau', ext_potential=None ,disp=lambda x: 10, Nmax=10, ftoll=0.001, flarefit='pol',nproc=1,toll=1E-4,outdir='gasHeigth'):
    """
    #Questo programma calcola l'altezza scala di un disco gassoso date tre componenti:
    1-un alone di DM (iso o NFW)
    par:
        -hlaw 'iso' or 'nfw' for pseudo-isothermal law or nfw
                --iso d(r): d0(r)/(1+(r/rc)^2)
                --nfw d(r): d0(r)/((r/rs)*(1+(r/rs))^2) where the link the classical NFW variables are d0=deltac dcrit (eq. 3.3 my thesis)
        -d0 Central density of the halo in msol/kpc^3 for iso or typical density for nfw
        -rc Core radius for isothermal law or scale radius rs for nfw
        -e Halo ellipticity e*e=1-q*q  where q=b/a that is the axis ratio (e=0 for spherical halo)
        -rcut, Radius where d drop to 0
    2-un disco stellare
        -s_sigma0= Deprojected central surface density
        -s_rd= Exponential Radial scale length
        -s_zlaw= Vertical density law (gau for gaussian, sech for sech(x)^2, exp for exponential)
        -s_zd=disc height scale (zd for sech and exp, sigma for gaussian)
    3-disco gassoso
        vedi sotto

    Per quanto riguarda l'alone l integrazione viene effettuata seguento eq. 2.88 (BT), per quanto riguarda i dischi viene usata un estensione
    delle equazioni trovate sul Cuddeford, 1993.


    :param R: Array with the Radial coordinate where to calc the potential
    :param Z: Array with the Vertical coordinate where to calc the potential
    :param hlaw: 'iso' or 'nfw' for pseudo-isothermal law or nfw
                --iso d(r): d0(r)/(1+(r/rc)^2)
                --nfw d(r): d0(r)/((r/rs)*(1+(r/rs))^2) where the link the classical NFW variables are d0=deltac dcrit (eq. 3.3 my thesis)
    :param d0: Central density of the halo in msol/kpc^3 for iso or typical density for nfw. If None the halo component will not be used
    :param rc: Core radius for isothermal law or scale radius rs for nfw
    :param e: Halo ellipticity e*e=1-q*q  where q=b/a that is the axis ratio (e=0 for spherical halo)
    :param rcut: Radius where d of the halo drop to 0
    :param s_sigma0: Deprojected central surface density of the star disk [Msun/kpc^2] If None the stellar component will not be used
    :param s_rd: Exponential Radial scale length [kpc]
    :param s_zd: disc height scale (zd for sech and exp, sigma for gaussian) [kpc]
    :param s_rcut: Cut Radius for the radial integration [kpc]
    :param s_zcut: Z cut for the vertical integration [kpc]
    :param s_zlaw:  Vertical density law (gau for gaussian, sech for sech(x)^2, exp for exponential)
    :param g_sigma0: Deprojected central surface density of the gaseous disk [Msun/kpc^2] If None the gaseous component will not be used
    :param g_rcut: Cut Radius for the radial integration [kpc]
    :param g_zcut: Z cut for the vertical integration [kpc]
    :param g_rlaw: Function of one variable describing the surface density law of the HI disk
    :param g_zlaw: Function of one variable describing the radial behaviour of the height scale
	:param ext_potential: External potential to be inclueded in the form of an arry with col-0:R col-1:Z col_2=Pot in
    :param disp: Function of one variable describing the radial behaviour of the velocity dispersion [km/s]
    :param Nmax: Max number of iteration (def: 10 )
    :param ftoll: Tollerance to achieve (def: 0.001 kpc)
    :param flarefit: Method to be used to find a functional form for the flaring in the routine. It can be:
            -'pol': Use a 4-order polynomial
            -'int': Interpolation of order 2
    :param nproc: Number of multiprocces to be used
    :param toll: Abs and Rel tollerance for the integration (see scipy.nquad)
    :param outdir: Name of the directory where to put the output
    :return: A numpy array with the final HI scale height for every radius of the input table R and the last fitted functional form and the derivative of the functional form
    """

    #Make dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    #################################MAIN######################################
    nrow=len(R)*len(Z) #Number or processed row

    ##Initialize Parallel
    pardo=ParDo(nproc)
    #pardo=ts.ParDo(nproc)

    #----------------FIxed potential---------------#
    #Halo
    if d0!=None: #If halo is present
        def halo(R):#Def halo potential routine
            pardo.output.put(cpot_halo(R,Z,d0,rc,e,rcut,hlaw=hlaw,toll=toll))

        htab=pardo.run(R,halo) #Potential tab for the halo

        np.savetxt(outdir+'/'+'halo_pot.txt',htab) #Save the potential tab

    #Stellar disk
    if s_sigma0!=None:
    #radialdens= stf.StringFunction('exp(-x/4)')
        radialdens= lambda r: np.exp(-r/s_rd) #Def surface density law
        flarelaw= lambda r: s_zd #Def flarelaw

        def sdisk(R): #Def star potential routine
            pardo.output.put(cpot_disk(R,Z, s_sigma0,s_zlaw,radialdens,flarelaw,s_rcut,s_zcut,toll))

        stab=pardo.run(R,sdisk)  #Potential tab for the star
        np.savetxt(outdir+'/'+'stardisk_pot.txt',stab) #Save the potential tab

	#External
    if ext_potential!=None: np.savetxt(outdir+'/'+'ext_pot.dat',ext_potential) #save the external potential tab


    #Total Fixed potential
    if (d0!=None) & (s_sigma0!=None):
        ptot=stab
        if ext_potential!=None: ptot[:,2]=stab[:,2]+htab[:,2] + ext_potential[:,2]
        else: ptot[:,2]=stab[:,2]+htab[:,2]
        np.savetxt(outdir+'/'+'pot_tot.txt',ptot)
    elif (d0!=None):
        ptot=htab
        if ext_potential!=None: ptot[:,2]+= ext_potential[:,2]
        np.savetxt(outdir+'/'+'pot_tot.txt',ptot)
    elif (s_sigma0!=None):
        ptot=stab
        if ext_potential!=None: ptot[:,2]+= ext_potential[:,2]
        np.savetxt(outdir+'/'+'pot_tot.txt',ptot)
    elif (ext_potential!=None):
        ptot=ext_potential
        np.savetxt(outdir+'/'+'pot_tot.txt',ptot)
    #------------------------------------------------#
    #Calc initial flaring for gaseous disk
    dens=cdens(ptot,disp)
    np.savetxt(outdir+'/'+'dens.dat',dens)
    if g_zlaw=='sech': g_zlaw_h='sech2'
    if g_zlaw=='gau': g_zlaw_h='gau'
    tabf,tabhw,c=ft.fitzprof(dens[:][:,[0,1,3]],dists=(g_zlaw_h), outdir=outdir+'/'+'run_0',plot=True,diagnostic=True,output=True)
    #------------------------------------------------#
    N=0
    #Gaseous disk
    if g_sigma0!=None:
        #If pol fitt polynomial
        if flarefit=='pol':
            z=np.polyfit(tabf[:,0],tabf[:,1],4)
            print(z)
            if z[-1]==0: print('Run '+ str(N) +' Warning flare at R=0 is negative')
            p0=z[0]
            p1=z[1]
            p2=z[2]
            p3=z[3]
            p4=z[4]
            flarelaw= lambda x: p0*x*x*x*x + p1*x*x*x +p2*x*x + p3*x + p4
            #ad un certo punto devo affrontare questa questione
            #flarelaw= lambda x: np.where(x<=R[-1],p0*x*x*x*x + p1*x*x*x +p2*x*x + p3*x + p4,p0*R[-1]*R[-1]*R[-1]*R[-1] + p1*R[-1]*R[-1]*R[-1] +p2*R[-1]*R[-1] + p3*R[-1]+ p4)
        elif flarefit=='int':
            flarelaw= sp.InterpolatedUnivariateSpline(tabf[:,0],tabf[:,1],ext=3,k=2)
        else: raise IOError('Wrong or missed flarefit, use pol for polynomial fit or int for interpolation')


        #Gaseous disk
        N=0
        diffmax=ftoll+2
        while (N<Nmax) and (diffmax>ftoll):

            oldtabf=tabf


            fig10=plt.figure()
            ax10=fig10.add_subplot(1,1,1)
            ax10.scatter(tabf[:,0],tabf[:,1])
            ax10.plot(tabf[:,0],flarelaw(tabf[:,0]),label='Guess')


            def gdisk(R):
                pardo.output.put(cpot_disk(R,Z, g_sigma0,g_zlaw,g_rlaw,flarelaw,g_rcut,g_zcut,toll))

            gtab=pardo.run(R,gdisk)
            np.savetxt(outdir+'/gtab'+'run_'+str(N+1)+'.dat',gtab)
            htabtmp=gtab
            htabtmp[:,2]=gtab[:,2]+ptot[:,2]

            dens=cdens(htabtmp,disp)
            if g_zlaw=='sech': g_zlaw_h='sech2'
            if g_zlaw=='gau': g_zlaw_h='gau'
            tabf,tabhw,c=ft.fitzprof(dens[:][:,[0,1,3]],dists=(g_zlaw_h),plot=True,diagnostic=True,outdir=outdir+'/'+'run_'+str(N+1),output=True)

            if flarefit=='pol':
                 z=np.polyfit(tabf[:,0],tabf[:,1],4)
                 print(z)
                 if z[-1]==0: print('Run '+ str(N) +' Warning flare at R=0 is negative')
                 p0=z[0]
                 p1=z[1]
                 p2=z[2]
                 p3=z[3]
                 p4=z[4]
                 flarelaw= lambda x: p0*x*x*x*x + p1*x*x*x +p2*x*x + p3*x + p4
                 flarelaw_der= lambda x: 4*p0*x*x*x + 3*p1*x*x + 2*p2*x + p3
            elif flarefit=='int':
                 flarelaw= sp.InterpolatedUnivariateSpline(tabf[:,0],tabf[:,1],ext=3,k=2)
                 flarelaw_der=flarelaw.derivative()


            diffmax=np.max(np.abs(tabf[:,1]-oldtabf[:,1]))

            ax10.scatter(tabf[:,0],tabf[:,1],label='Results',c='r')
            plt.legend()
            fig10.savefig(outdir+'/'+'run_'+str(N+1)+'/flarefit.pdf')

            N+=1
            print('N',N,'Nmax',Nmax)
            print('diffmax',diffmax,'Tollerance',ftoll)

    return tabf,flarelaw,flarelaw_der

def plotvel(tabvel,outdir=os.getcwd(),outname='vel.pdf',vdm_low=None,vdm_up=None,vt_low=None,vt_up=None):
    fig=plt.figure()
    x=tabvel[:,0]
    if tabvel.shape[1]==6:
        plt.scatter(x,tabvel[:,-1],label='Vobs')
    elif tabvel.shape[1]==7:
        plt.errorbar(x,tabvel[:,-2],yerr=tabvel[:,-1],fmt=' ', color='b',marker='o',markersize=3)
    else: raise ValueError()
    plt.plot(x,tabvel[:,1],label='Vstar',c='red')
    plt.plot(x,tabvel[:,2],label='Vgas',c='darkgreen')
    plt.plot(x,tabvel[:,3],label='Vhalo',c='b')
    plt.plot(x,tabvel[:,4],label='Vtot',c='black')

    if vdm_low!=None:
        plt.fill_between(x,vdm_low,vdm_up,color='blue',alpha=0.4)
        plt.fill_between(x,vt_low,vt_up,color='gold',alpha=0.6)

    plt.legend(loc='upper center',ncol=5,bbox_to_anchor=(0.5, 1.1))
    plt.xlabel('R [kpc]')
    plt.ylabel('Vc [km/s]')
    plt.xlim(-np.min(x)*1.05,np.max(x)*1.05)
    plt.ylim(-5,np.max(tabvel[:,5])*1.05)
    plt.grid()
    fig.savefig(outdir+'/'+outname)


def fit_error(x,data,func,fit_par,m=None,yerr=None):
    n=len(data)
    if m==None: m=3*n
    res=np.zeros(shape=(m,len(fit_par)))
    for i in range(m):
        index=np.random.randint(0,n,n)
        fitdata=data[index]
        xfit=x[index]
        popt,pcov=curve_fit(func,xfit,fitdata,p0=fit_par,absolute_sigma=True,sigma=yerr)
        #res.append(popt)
        for k in range(len(popt)):
            res[i,k]=popt[k]
    err=np.sqrt(np.sum((res-fit_par)*(res-fit_par),axis=0)/(m))
    return np.array(res),err

def printf(*args):
    print(*args)
    sys.stdout.flush()

def lnprob_iso(par,r,vel,velerr):
    d0,rc=par
    #Log likelihood
    model=rot_iso(r,np.abs(d0),np.abs(rc))
    inv_sigma2=1.0/((velerr*velerr))
    lnlike=-0.5*(np.sum((vel-model)**2*inv_sigma2 - np.log(inv_sigma2)))
    #Log prior
    if (1e2<d0<1e9) and (0.001<rc<5000): lnprior=0.0 #Uniform prior log(1)=0
    else: lnprior=-np.inf #log(0)=-inf
    return lnlike+lnprior #Posterior (log(lik*prior)=log(lik)+log(prior))

def lnprob_nfw(par,r,vel,velerr):
    c,v200=par
    #Log likelihood
    model=rot_nfw(r,np.abs(c),np.abs(v200))
    inv_sigma2=1.0/((velerr*velerr))
    lnlike=-0.5*(np.sum((vel-model)**2*inv_sigma2 - np.log(inv_sigma2)))
    #Log prior
    if (1e-4<c<1e3) and (1e-3<v200<1e3): lnprior=0.0 #Uniform prior log(1)=0
    else: lnprior=-np.inf #log(0)=-inf
    posterior=lnlike+lnprior
    if not np.isfinite(posterior): return -np.inf
    else: return posterior #Posterior (log(lik*prior)=log(lik)+log(prior))

