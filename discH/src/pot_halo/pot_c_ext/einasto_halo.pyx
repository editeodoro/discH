#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin, pow, exp
from .general_halo cimport m_calc, potential_core, integrand_core, vcirc_core
from scipy.integrate import quad
from scipy._lib._ccallback import LowLevelCallable
import numpy as np
cimport numpy as np
ctypedef double * double_ptr
ctypedef void * void_ptr
from cython_gsl cimport *

cdef double PI=3.14159265358979323846


cdef double dn_func(double n) nogil:

    cdef:
        double n2=n*n
        double n3=n2*n
        double n4=n3*n
        double a0=3*n
        double a1=-1./3.
        double a2=8./(1215.*n)
        double a3=184./(229635.*n2)
        double a4=1048/(31000725.*n3)
        double a5=-17557576/(1242974068875.*n4)

    return a0+a1+a2+a3+a4+a5

cdef double dens_einasto(double m, void * params) nogil:

    cdef:
        double d0 = (<double_ptr> params)[0]
        double rs = (<double_ptr> params)[1]
        double n  = (<double_ptr> params)[2]
        double dn=dn_func(n)
        double x=m/rs
        double ee=dn*pow(x,1/n)

    return 2.*m*d0*exp(-ee)

cdef double psi_einasto(double d0, double rs, double n, double m,  double toll) nogil:

    cdef:
        double result, error
        gsl_integration_workspace * w
        gsl_function F
        double params[3]

    params[0] = d0
    params[1] = rs
    params[2] = n

    W = gsl_integration_workspace_alloc (1000)


    F.function = &dens_einasto
    F.params = params


    gsl_integration_qag(&F, 0, m, toll, toll, 1000, GSL_INTEG_GAUSS15, W, &result, &error)
    gsl_integration_workspace_free(W)

    return result

cdef double integrand_einasto(int nn, double *data) nogil:


    cdef:
        double m = data[0]

    if m==0.: return 0 #Xi diverge to infinity when m tends to 0, but the integrand tends to 0

    cdef:
        double R = data[1]
        double Z = data[2]
        double mcut = data[3]
        double d0 = data[4]
        double rs = data[5]
        double n = data[6]
        double e = data[7]
        double toll = data[8]
        double psi, result #, num, den

    if (m<=mcut): psi=psi_einasto(d0, rs, n, m, toll)
    else: psi=psi_einasto(d0, rs, n, mcut, toll)

    result=integrand_core(m, R, Z, e, psi)
    #num=xi(m,R,Z,e)*(xi(m,R,Z,e)-e*e)*sqrt(xi(m,R,Z,e)-e*e)*m*psi
    #den=((xi(m,R,Z,e)-e*e)*(xi(m,R,Z,e)-e*e)*R*R)+(xi(m,R,Z,e)*xi(m,R,Z,e)*Z*Z)

    return result

cdef double  _potential_einasto(double R, double Z, double mcut, double d0, double rs, double n, double e, double toll):


    cdef:
        double G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
        double cost=2*PI*G
        double m0
        double psi
        double intpot
        double result

    m0=m_calc(R,Z,e)

    #Integ
    import discH.src.pot_halo.pot_c_ext.einasto_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_einasto')


    intpot=quad(fintegrand,0.,m0,args=(R,Z,mcut,d0,rs,n,e,toll),epsabs=toll,epsrel=toll)[0]


    psi=psi_einasto(d0,rs,n,mcut,toll)

    result=potential_core(e, intpot, psi)

    return result



cdef double[:,:]  _potential_einasto_array(double[:] R, double[:] Z, int nlen, double mcut, double d0, double rs, double n, double e, double toll):


    cdef:
        double G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
        double cost=2*PI*G
        double m0
        double psi
        double[:,:] ret=np.empty((nlen,3), dtype=np.dtype("d"))
        double intpot
        int i



    #Integ
    import discH.src.pot_halo.pot_c_ext.einasto_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_einasto')


    for  i in range(nlen):


        ret[i,0]=R[i]
        ret[i,1]=Z[i]

        m0=m_calc(R[i],Z[i],e)

        intpot=quad(fintegrand,0.,m0,args=(R[i],Z[i],mcut,d0,rs,n,e,toll),epsabs=toll,epsrel=toll)[0]

        psi=psi_einasto(d0,rs,n, mcut, toll)

        ret[i,2]=potential_core(e, intpot, psi)


    return ret



cdef double[:,:]  _potential_einasto_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, double mcut, double d0, double rs, double n,  double e, double toll):


    cdef:
        double G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
        double cost=2*PI*G
        double m0
        double psi
        double[:,:] ret=np.empty((nlenR*nlenZ,3), dtype=np.dtype("d"))
        double intpot
        int i, j, c



    #Integ
    import discH.src.pot_halo.pot_c_ext.einasto_halo as mod
    fintegrand=LowLevelCallable.from_cython(mod,'integrand_einasto')

    c=0
    for  i in range(nlenR):
        for j in range(nlenZ):

            ret[c,0]=R[i]
            ret[c,1]=Z[j]

            m0=m_calc(R[i],Z[j],e)

            intpot=quad(fintegrand,0.,m0,args=(R[i],Z[j],mcut,d0,rs,n,e,toll),epsabs=toll,epsrel=toll)[0]

            psi=psi_einasto(d0,rs,n,mcut,toll)

            ret[c,2]=potential_core(e, intpot, psi)
            #if (e<=0.0001):
            #    ret[c,2] = -cost*(psi-intpot)
            #else:
            #    ret[c,2] = -cost*(sqrt(1-e*e)/e)*(psi*asin(e)-e*intpot)

            c+=1

    return ret


cpdef potential_einasto(R, Z, d0, rs, n, e, mcut, toll=1e-4, grid=False):

    if isinstance(R, float) or isinstance(R, int):
        if isinstance(Z, float) or isinstance(Z, int):
            return np.array(_potential_einasto(R=R,Z=Z,mcut=mcut,d0=d0, rs=rs, n=n, e=e,toll=toll))
        else:
            raise ValueError('R and Z have different dimension')
    else:
        if grid:
            R=np.array(R,dtype=np.dtype("d"))
            Z=np.array(Z,dtype=np.dtype("d"))
            return np.array(_potential_einasto_grid( R=R, Z=Z, nlenR=len(R), nlenZ=len(Z), mcut=mcut, d0=d0, rs=rs, n=n, e=e,toll=toll))
        elif len(R)==len(Z):
            nlen=len(R)
            R=np.array(R,dtype=np.dtype("d"))
            Z=np.array(Z,dtype=np.dtype("d"))
            return np.array(_potential_einasto_array( R=R, Z=Z, nlen=len(R), mcut=mcut, d0=d0, rs=rs, n=n, e=e,toll=toll))
        else:
            raise ValueError('R and Z have different dimension')