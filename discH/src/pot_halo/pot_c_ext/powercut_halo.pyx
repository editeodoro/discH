#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin, pow, exp
from .general_halo cimport m_calc, potential_core, integrand_core, vcirc_core, vcirc_norm
from scipy.integrate import quad 
from scipy._lib._ccallback import LowLevelCallable
from scipy.special.cython_special cimport gammaincc, gamma
import numpy as np
cimport numpy as np

cdef double psi_powercut(double d0, double rs, double rb, double alpha, double m, double toll) nogil:

    return -d0*rb*rb*pow(m/rs,-alpha)*pow(m*m/(rb*rb),alpha/2.)*gammaincc(1-alpha/2.,m*m/(rb*rb))*gamma(1-alpha/2.)


cdef double integrand_powercut(int nn, double *data) nogil:

    cdef:
        double m = data[0]
        double R = data[1]
        double Z = data[2]
        double mcut = data[3]
        double d0 = data[4]
        double rs = data[5]
        double rb = data[6]
        double alpha = data[7]
        double e = data[9]
        double toll = data[10]
        double psi, result

    if m==0.: return 0 #Xi diverge to infinity when m tends to 0, but the integrand tends to 0    

    if (m<=mcut): psi=psi_powercut(d0, rs, rb, alpha, m, toll)
    else: psi=psi_powercut(d0, rs, rb, alpha, mcut, toll)

    result=integrand_core(m, R, Z, e, psi)

    return result


cdef double  _potential_powercut(double R, double Z, double mcut, double d0, double rs, double rb, double alpha, double e, double toll):

    cdef:
        double m0
        double psi
        double intpot
        double result

    m0=m_calc(R,Z,e)

    #Integ
    import discH.src.pot_halo.pot_c_ext.powercut_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_powercut')
    intpot=quad(fintegrand,0.,m0,args=(R,Z,mcut,d0,rs,rb,alpha,e,toll),epsabs=toll,epsrel=toll)[0]
    psi=psi_powercut(d0,rs,rb,alpha,mcut,toll)
    result=potential_core(e, intpot, psi)

    return result



cdef double[:,:] _potential_powercut_array(double[:] R, double[:] Z, int nlen, double mcut, double d0, double rs, double rb, double alpha, double e, double toll):

    cdef:
        double[:,:] ret=np.empty((nlen,3), dtype=np.dtype("d"))
        int i

    for  i in range(nlen):
        ret[i,0]=R[i]
        ret[i,1]=Z[i]
        ret[i,2]=_potential_powercut(R[i],Z[i],mcut,d0,rs,rb,alpha,e,toll)
    
    return ret



cdef double[:,:]  _potential_powercut_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, double mcut, double d0, double rs, double rb, double alpha, double e, double toll):
    
    cdef:
        double[:,:] ret=np.empty((nlenR*nlenZ,3), dtype=np.dtype("d"))
        int i, j, c
        
    c=0
    for  i in range(nlenR):
        for j in range(nlenZ):
            ret[c,0]=R[i]
            ret[c,1]=Z[j]
            ret[c,2]=_potential_powercut(R[i],Z[j],mcut,d0,rs,rb,alpha,e,toll)
            c+=1

    return ret


cpdef potential_powercut(R, Z, d0, rs, rb, alpha, e, mcut, toll=1e-4, grid=False):

    if isinstance(R, float) or isinstance(R, int):
        if isinstance(Z, float) or isinstance(Z, int):
            return np.array(_potential_powercut(R=R,Z=Z,mcut=mcut,d0=d0,rs=rs,rb=rb,alpha=alpha,e=e,toll=toll))
        else:
            raise ValueError('R and Z have different dimension')
    else:
        if grid:
            R=np.array(R,dtype=np.dtype("d"))
            Z=np.array(Z,dtype=np.dtype("d"))
            return np.array(_potential_powercut_grid(R=R,Z=Z,nlenR=len(R),nlenZ=len(Z),mcut=mcut,d0=d0,rs=rs,rb=rb,alpha=alpha,e=e,toll=toll))
        elif len(R)==len(Z):
            nlen=len(R)
            R=np.array(R,dtype=np.dtype("d"))
            Z=np.array(Z,dtype=np.dtype("d"))
            return np.array(_potential_powercut_array(R=R,Z=Z,nlen=len(R),mcut=mcut,d0=d0,rs=rs,rb=rb,alpha=alpha,e=e,toll=toll))
        else:
            raise ValueError('R and Z have different dimension')

#####################################################################
#Vcirc
cdef double vcirc_integrand_powercut(int n, double *data) nogil:
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
        double dens, core
        
    core = vcirc_core(m, R, e)
    dens = pow(m/rs,-alpha)*exp(-m*m/(rb*rb))
    
    return core*dens



cdef double _vcirc_powercut(double R, double d0, double rs, double rb, double alpha, double e, double toll):
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
        double intvcirc
        double result

    #Integ
    import discH.src.pot_halo.pot_c_ext.powercut_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'vcirc_integrand_powercut')
    intvcirc=quad(fintegrand,0.,R,args=(R,rs,rb,alpha,e),epsabs=toll,epsrel=toll)[0]

    return vcirc_norm(intvcirc,d0,e)


cdef double[:,:] _vcirc_powercut_array(double[:] R, int nlen, double d0, double rs, double rb, double alpha, double e, double toll):
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
        int i
        double[:,:] ret=np.empty((nlen,2), dtype=np.dtype("d"))

    for  i in range(nlen):
        ret[i,0]=R[i]
        ret[i,1]=_vcirc_powercut(R[i],d0,rs,rb,alpha,e,toll)

    return ret


cpdef vcirc_powercut(R, d0, rs, rb, alpha, e, toll=1e-4):
    """ Calculate the Vcirc on the plane 

    :return: 2-col array:
        0-R
        1-Vcirc(R)
    """

    if isinstance(R, (int,float)):
        ret = 0. if R==0 else _vcirc_powercut(R,d0,rs,rb,alpha,e,toll)
    else:
        ret=_vcirc_powercut_array(R,len(R),d0,rs,rb,alpha,e,toll)
        ret[:,1]=np.where(R==0,0,ret[:,1])

    return ret