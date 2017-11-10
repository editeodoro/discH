#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin, pow, exp
from .general_triaxial_halo cimport integrand_core, potential_core # ,vcirc_core
from scipy.integrate import quad
from scipy._lib._ccallback import LowLevelCallable
from scipy.special.cython_special cimport gammaincc, gamma
import numpy as np
cimport numpy as np
from cython_gsl cimport *

cdef double PI=3.14159265358979323846

cdef double psi_triaxial_exponential(double d0, double rs, double alpha, double m) nogil:
    
    """ Return psi(m) = \int_0^m**2 (rho(m**2) dm**2). See BT08, eq 2.117 """
    
    return -2.*d0*m*m*pow(pow(m/rs,alpha),-2./alpha)*gammaincc(2./alpha,pow(m/rs,alpha))*gamma(2./alpha)/alpha



cdef double integrand_triaxial_exponential(int nn, double *data) nogil:
    
    """ Returns the integrand function for eq 2.140 BT08, but in ds:
         
           integ = -2*psi(m)-psi(inf)/sqrt((1+s*s*(a*a-1))*(1+s*s*(b*b-1))*(1+s*s*(c*c-1)))
           where s = 1 / sqrt(1+tau)
    """
    
    cdef:
        double s = data[0]
        double x = data[1]
        double y = data[2]
        double z = data[3]
        double mcut = data[4]
        double d0 = data[5]
        double rs = data[6]
        double alpha = data[7]
        double a = data[8]
        double b = data[9]
        double c = data[10]
        double toll = data[11]
        double m, tau, psi, psicut

    tau = 1/(s*s) - 1
    m  = a*sqrt(x*x/(tau+a*a)+y*y/(tau+b*b)+z*z/(tau+c*c))

    if m==0.: return 0 
    
    # Psi at inf (mcut)
    psicut = psi_triaxial_exponential(d0, rs, alpha, mcut)
    # Psi at m
    if (m<=mcut): psi=psi_triaxial_exponential(d0, rs, alpha, m)
    else: psi=psicut
    
    return integrand_core(s,a,b,c,psi,psicut)



cdef double _potential_triaxial_exponential(double x, double y, double z, double mcut, double d0, double rs, double alpha, double a, double b, double c, double toll):
    """ Returns the potential at (x,y,z) from eq. 2.140 BT08 """
    
    cdef:
        double intpot

    #Integral
    import discH.src.pot_halo.pot_c_ext.triaxial_exponential_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_triaxial_exponential')
    
    # For convenience, we integrate in the variable s = 1/sqrt(tau +1) 
    intpot=quad(fintegrand,0.,1.,args=(x,y,z,mcut,d0,rs,alpha,a,b,c,toll),epsabs=toll,epsrel=toll)[0]
    
    return potential_core(a,b,c,intpot)



cdef double[:,:] _potential_triaxial_exponential_array(double[:] x, double[:] y, double[:] z, int nlen, double mcut, double d0, double rs, double alpha, double a, double b, double c, double toll):

    cdef:
        double[:,:] ret=np.empty((nlen,4), dtype=np.dtype("d"))
        int i
    
    for i in range(nlen):
        ret[i,0]=x[i]
        ret[i,1]=y[i]
        ret[i,2]=z[i]
        ret[i,3]= _potential_triaxial_exponential(x[i],y[i],z[i],mcut,d0,rs,alpha,a,b,c,toll)

    return ret



cdef double[:,:] _potential_triaxial_exponential_grid(double[:] x, double[:] y, double[:] z, int nlenx, int nleny, int nlenz, double mcut, double d0, double rs, double alpha, double a, double b, double c, double toll):
    cdef:
        double[:,:] ret=np.empty((nlenx*nleny*nlenz,4), dtype=np.dtype("d"))
        int i, j, k, l
 
    #Integ
    import discH.src.pot_halo.pot_c_ext.triaxial_exponential_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_triaxial_exponential')

    l=0
    for k in range(nlenz):
        for j in range(nleny):
            for  i in range(nlenx):
                ret[l,0]=x[i]
                ret[l,1]=y[j]
                ret[l,2]=z[k]                
                ret[l,3]=_potential_triaxial_exponential(x[i],y[j],z[k],mcut,d0,rs,alpha,a,b,c,toll) 
                l+=1

    return ret



cpdef potential_triaxial_exponential(x, y, z, d0, rs, alpha, a, b, c, mcut, toll=1e-4, grid=False):

    if isinstance(x, (float,int)) and isinstance(y,(float,int)) and isinstance(z, (float,int)):
        return np.array(_potential_triaxial_exponential(x=x,y=y,z=z,mcut=mcut,d0=d0,rs=rs,alpha=alpha,a=a,b=b,c=c,toll=toll))
    else:
        if grid:
            x=np.array(x,dtype=np.dtype("d"))
            y=np.array(y,dtype=np.dtype("d"))
            z=np.array(z,dtype=np.dtype("d"))
            return np.array(_potential_triaxial_exponential_grid(x=x,y=y,z=z,nlenx=len(x),nleny=len(y),nlenz=len(z),mcut=mcut,d0=d0,rs=rs,alpha=alpha,a=a,b=b,c=c,toll=toll))
        elif len(x)==len(y)==len(z):
            nlen=len(x)
            x=np.array(x,dtype=np.dtype("d"))
            y=np.array(y,dtype=np.dtype("d"))
            z=np.array(z,dtype=np.dtype("d"))
            return np.array(_potential_triaxial_exponential_array(x=x,y=y,z=z,nlen=len(x),mcut=mcut,d0=d0,rs=rs,alpha=alpha,a=a,b=b,c=c,toll=toll))
        else:
            raise ValueError('x, y and z have different dimension')


''' To be implemented
#####################################################################
#Vcirc
cdef double vcirc_integrand_triaxial_exponential(int n, double *data) nogil:
    """
    Integrand function for vcirc  on the plane (Eq. 2.132 in BT2)
    """

    cdef:
        double m      = data[0]
        double R      = data[1]
        double rs     = data[2]
        double rb     = data[3]
        double alpha  = data[4]
        double e      = data[6]
        double dens
        double core

    core = vcirc_core(m, R, e)
    dens = 1                 # NEED to CHANGE THIS !!!!!!!!!!!!!!!!!

    return core*dens



cdef double _vcirc_triaxial_exponential(double R, double d0, double rs, double rb, double alpha, double e, double toll):
    """
    Calculate Vcirc on a single point on the plane
    :param R: radii array (kpc)
    :param d0: Central density (Msol/kpc^3)
    :param rs: scale radius, radius where the mass is the half of the total (kpc)
    :param n:
    :param e: ellipticity
    :return:
    """

    cdef:
        double G=4.302113488372941e-06 #G constant in  kpc km2/(msol s^2)
        double cost=4*PI*G*sqrt(1-e*e)*d0
        double norm
        double intvcirc
        double result

    #Integ
    import discH.src.pot_halo.pot_c_ext.triaxial_exponential_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'vcirc_integrand_triaxial_exponential')
    intvcirc=quad(fintegrand,0.,R,args=(R,rs,rb,alpha,e),epsabs=toll,epsrel=toll)[0]
    result=sqrt(cost*intvcirc)

    return result


cdef double[:,:] _vcirc_triaxial_exponential_array(double[:] R, int nlen, double d0, double rs, double rb, double alpha, double e, double toll):
    """
    Calculate Vcirc on a single point on the plane
    :param R: radii array (kpc)
    :param d0: Central density (Msol/kpc^3)
    :param rs: scale radius, radius where the mass is the half of the total (kpc)
    :param n:
    :param e: ellipticity
    :return:
    """

    cdef:
        double G=4.302113488372941e-06 #G constant in  kpc km2/(msol s^2)
        double cost=4*PI*G*sqrt(1-e*e)*d0
        double intvcirc
        int i
        double[:,:] ret=np.empty((nlen,2), dtype=np.dtype("d"))

    #Integ
    import discH.src.pot_halo.pot_c_ext.triaxial_exponential_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'vcirc_integrand_triaxial_exponential')

    for  i in range(nlen):
        ret[i,0]=R[i]
        intvcirc=quad(fintegrand,0.,R[i],args=(R[i],rs,rb,alpha,e),epsabs=toll,epsrel=toll)[0]
        ret[i,1]=sqrt(cost*intvcirc)

    return ret


cpdef vcirc_triaxial_exponential(R, d0, rs, rb, alpha, e, toll=1e-4):
    """ Calculate the Vcirc on the plane 

    :return: 2-col array:
        0-R
        1-Vcirc(R)
    """

    if isinstance(R, float) or isinstance(R, int):
        if R==0: ret=0
        else: ret= _vcirc_triaxial_exponential(R,d0,rs,rb,alpha,e,toll)
    else:
        ret=_vcirc_triaxial_exponential_array(R,len(R),d0,rs,rb,alpha,e,toll)
        ret[:,1]=np.where(R==0,0,ret[:,1])

    return ret
            
'''