#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
from cython.parallel cimport prange
import scipy.special as sf
import scipy.interpolate as sp
from scipy.integrate import quad, nquad, fixed_quad
import functools
import fermi.tsintegrator as ft
import multiprocessing as mp
import matplotlib.pyplot as plt
#import fitlib as ft
import os
import sys
from scipy.optimize import curve_fit
import emcee
import datetime
import time
from libc.math cimport sqrt, asin
import numpy as np
cimport numpy as np
from scipy._lib._ccallback import LowLevelCallable
from libc.stdio cimport printf
from numpy.ctypeslib import ndpointer


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
    """Auxiliary functions

    :param m: elliptical radius (m=sqrt(R^2+Z^2/q^2))
    :param R: cylindrical Radius
    :param Z: cylindrical height
    :param e: ellipticity (e-0 Spherical system, e=1 maximal flattening)
    :return: auxiliary functions for integrations
    """
    return  (R*R+Z*Z+e*e*m*m+np.sqrt((e*m)**4-2*e*e*m*m*(R*R-Z*Z)+(R*R+Z*Z)*(R*R+Z*Z)))/(2*m*m)

cpdef double psi_iso(double d0,double rc,double m):
    """Auxiliary functions linked to density law iso:
    d=d0/(1+m/rc^2)

    :param d0: Central density at (R,Z)=(0,0) [Msol/kpc^3]
    :param rc: Core radius [Kpc]
    :param m: spheroidal radius
    :return:
    """
    return d0*rc*rc*(np.log(1+((m*m)/(rc*rc))))

cpdef double psi_nfw(double d0,double rc,double m):
    """Auxiliary functions linked to density law nfw:
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
    """ Potential integrand for isothermal halo: d=d0/(1+m/rc^2)

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
    """Potential integrand for isothermal halo: d=d0/((m/rc)*(1+m/rc)^2)

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
    """Calculate the potential of a isothermal (spheroid d=d0/(1+m/rc^2))
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

cpdef double potentialh_iso2(double R,double Z,double d0, double rc,double e,double rcut,double toll):
    """Calculate the potential of a isothermal (spheroid d=d0/(1+m/rc^2))
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
    print('we')
    G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
    cost=2*np.pi*G
    mcut=rcut*np.sqrt(2)
    m0=np.sqrt(R*R+(Z*Z)/(1-e*e)) #Value of m at R,Z
    fint=functools.partial(integrandhiso,e=e,R=R,Z=Z,d0=d0,rc=rc,mcut=mcut)
    #intpot=quad(fint,0.,m0,epsabs=toll,epsrel=toll)[0]
    print('we2')
    f=ft.Tsintegrator1D(20,hstep=3)
    print('we3')
    print(fint(3))
    intpot=f.integrate(fint,0,m0)
    print(intpot)

    if (e<=0.0001): return -cost*(psi_iso(d0,rc,mcut)-intpot)
    else: return -cost*(np.sqrt(1-e*e)/e)*(psi_iso(d0,rc,mcut)*np.arcsin(e)-e*intpot)

cpdef double potentialh_nfw(double R, double Z, double d0, double rc, double e, double rcut, double toll):
    """Calculate the potential of a isothermal  d=d0/((m/rc)*(1+m/rc)^2)
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

cdef class Halo:

    cdef:
        double e
        double d0
        double rc
        double mcut

    def __cinit__(self,d0,rc,e,mcut=None):

        self.d0 = d0
        self.rc = rc
        self.e = e
        if mcut is None:
            self.mcut = 10*self.rc
        else:
            self.mcut = mcut

    cdef double m_calc(self,double R, double Z):

        cdef e=self.e

        return sqrt(R*R+Z*Z/(1-e*e))

    cdef double xi(self,double m,double R,double Z):
        """Auxiliary functions

        :param m: elliptical radius (m=sqrt(R^2+Z^2/q^2))
        :param R: cylindrical Radius
        :param Z: cylindrical height
        :param e: ellipticity (e-0 Spherical system, e=1 maximal flattening)
        :return: auxiliary functions for integrations
        """
        cdef e=self.e

        return  (R*R+Z*Z+e*e*m*m+np.sqrt((e*m)**4-2*e*e*m*m*(R*R-Z*Z)+(R*R+Z*Z)*(R*R+Z*Z)))/(2*m*m)



cdef class Isothermal_halo(Halo):
    "Isothermal halo class"


    def info(self):

        return self.d0, self.rc, self.e

    def get_xi(self,m,R,Z):

        a=self.xi(m,R,Z)

        return a

    cdef double info_c(self):

        return self.d0

    cdef double psi(self, double m):
        """Auxiliary functions linked to density law iso:
        d=d0/(1+m/rc^2)

        :param d0: Central density at (R,Z)=(0,0) [Msol/kpc^3]
        :param rc: Core radius [Kpc]
        :param m: spheroidal radius
        :return:
        """
        cdef:
            double d0=self.d0
            double rc=self.rc

        return d0*rc*rc*(np.log(1+((m*m)/(rc*rc))))

    cdef double integrand_c(self,double m, double R,double Z):
        """ Potential integrand for isothermal halo: d=d0/(1+m/rc^2)

        :param m: Spheroidal radius and integration variable
        :param mcut: Spheroidal radius where for m>mcut d=0
        :param R: Cylindrical coordinate
        :param Z: Cylindrical coordinate
        :param e: halo ellipticity
        :param d0: Central density
        :param rc: Core radius
        :return: integrand function
        """

        cdef:
            double mcut=self.mcut
            double e=self.e
            double xi, num, den



        if m==0:
            return 0. #Xi diverge to infinity when m tends to 0, but the integrand tends to 0
        else:
            if (m<=mcut): psi=self.psi(m)
            else: psi=psi=self.psi(mcut)
            xi=self.xi(m,R,Z)
            num=xi*(xi-e*e)*sqrt(xi-e*e)*m*psi
            den=((xi-e*e)*(xi-e*e)*R*R)+(xi*xi*Z*Z)
            return num/den


    cpdef integrand(self,m, R, Z):

        return self.integrand_c(m, R, Z)



    cdef double potential_single(self,double R,double Z,double toll):
        """Calculate the potential of a isothermal (spheroid d=d0/(1+m/rc^2))
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

        cdef:
            double G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
            double cost=2*np.pi*G
            double mcut= self.mcut
            double e= self.e
            double m0


        m0=self.m_calc(R,Z)

        intpot=quad(self.integrand,0.,m0,args=(R,Z),epsabs=toll,epsrel=toll)[0]

        if (e<=0.0001): return -cost*(self.psi(mcut)-intpot)
        else: return -cost*(sqrt(1-e*e)/e)*(self.psi(mcut)*asin(e)-e*intpot)

    cdef double potential_single2(self,double R,double Z,double toll):
        """Calculate the potential of a isothermal (spheroid d=d0/(1+m/rc^2))
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

        cdef:
            double G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
            double cost=2*np.pi*G
            double mcut= self.mcut
            double e= self.e
            double m0


        m0=self.m_calc(R,Z)
        f=ft.Tsintegrator1D(20,hstep=3)
        intpot=f.integrate(self.integrand,0.,m0,extra_args=(R,Z),use_c=False,use_array=False)

        if (e<=0.0001): return -cost*(self.psi(mcut)-intpot)
        else: return -cost*(sqrt(1-e*e)/e)*(self.psi(mcut)*asin(e)-e*intpot)

    cdef double[:] potential_array(self,double[:] R,double[:] Z,double toll, int nlen):

        cdef:
            int i
            double[:] ret_array=np.empty(nlen,dtype=np.float64)

        for i in range(nlen):
            ret_array[i]=self.potential_single(R[i], Z[i], toll)

        return ret_array

    cdef double[:,:] potential_grid(self,double[:] R,double[:] Z,double toll, int nlenR, int nlenZ):

        cdef:
            int i, j
            double[:,:] ret_array=np.empty((nlenR,nlenZ),dtype=np.float64)

        for i in range(nlenR):
            for j in range(nlenZ):
                ret_array[i,j]=self.potential_single(R[i], Z[j], toll)

        return ret_array

    cdef double[:] potential_array2(self,double[:] R,double[:] Z,double toll, int nlen):

        cdef:
            int i
            double[:] ret_array=np.empty(nlen,dtype=np.float64)

        for i in range(nlen):
            ret_array[i]=self.potential_single2(R[i], Z[i], toll)

        return ret_array

    def potential(self,R,Z,toll,grid=False):

        if isinstance(R, float) or isinstance(R, int):
            if isinstance(Z, float) or isinstance(Z, int):
                return self.potential_single(R, Z, toll)
            else:
                raise ValueError('R and Z have different dimension')
        else:
            if grid:
                R=np.array(R,dtype=np.dtype("d"))
                Z=np.array(Z,dtype=np.dtype("d"))
                return np.array(self.potential_grid( R, Z, toll,len(R),len(Z)))
            elif len(R)==len(Z):
                nlen=len(R)
                R=np.array(R,dtype=np.dtype("d"))
                Z=np.array(Z,dtype=np.dtype("d"))
                return np.array(self.potential_array( R, Z, toll,nlen))
            else:
                raise ValueError('R and Z have different dimension')

    def potential2(self,R,Z,toll):

        if isinstance(R, float) or isinstance(R, int):
            if isinstance(Z, float) or isinstance(Z, int):
                return self.potential_single2(R, Z, toll)
            else:
                raise ValueError('R and Z have different dimension')
        else:
            if len(R)==len(Z):
                nlen=len(R)
                R=np.array(R,dtype=np.dtype("d"))
                Z=np.array(Z,dtype=np.dtype("d"))
                return np.array(self.potential_array2( R, Z, toll,nlen))
            else:
                raise ValueError('R and Z have different dimension')

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
            #print('Now processing: %.3f %.3f' % (r,z))
            table[count,0]=r
            table[count,1]=z
            table[count,2]=pot(r,z,d0,rc,e,rcut,toll)
            count+=1
        countr=count
    return np.asarray(table)


cdef double test_func_c(int n , double *xx):

    #printf("%i",n)
    #n=2

    cdef:
        double x = xx[0]
        double R = xx[1]

    #printf("%f",xx)

    return x*R+asin(x/R)

def test_func(x,R):

    cdef:
        double[:] xx=np.array([x,R],dtype=np.dtype("d"),)
    #xx=np.array([x,R])
    n=2

    return test_func_c(n,&xx[0])


def integrate_test(xl,xu,R,Z):

    toll=1e-4
    intpot=quad(test_func,xl,xu,args=(R),epsabs=toll,epsrel=toll)[0]

    return intpot





import discH.src.cpotlib as mod

#mod.test_func_c


flow=LowLevelCallable.from_cython(mod,'test_func_c')


def integrate_test_sp(xl,xu,R,Z):

    toll=1e-4
    intpot=quad(flow,xl,xu,args=(R),epsabs=toll,epsrel=toll)[0]

    return intpot
