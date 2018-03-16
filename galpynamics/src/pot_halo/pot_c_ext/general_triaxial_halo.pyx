#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, asin

cdef double PI=3.14159265358979323846

cdef double integrand_core(double s, double a, double b, double c, double psi, double psicut) nogil:
    """
    Integrand core function for triaxial potential (eq 2.140 BT08, but in ds)
    :param s:       Integrand variable: s = 1 / sqrt(tau)
    :param a,b,c:   Axis ratios
    :param psi:     Psi function at m -> psi(m)
    :param psicut   Psi function at inf -> psi(mcut)
    
    :return:        integrand for eq. 2.140 BT08
    """
    return -2*(psi-psicut) / sqrt((1+s*s*(a*a-1))*(1+s*s*(b*b-1))*(1+s*s*(c*c-1)))

cdef double potential_core(double a, double b, double c, double intpot) nogil:
    """ Function to calcualte the potential (Use the formula 2.88b in BT 1987)

    :param a,b,c:   Axis ratios
    :param intpot:  Result of the integration
    :return:        Final potential
    """
    cdef:
        double G=4.498658966346282e-12 #G constant in  kpc^3/(msol Myr^2 )
        
    return -PI*G*b*c/a*intpot


''' To be implemented
cdef double vcirc_core(double m, double R, double e) nogil:
    """
    Core function to calculate the Vcirc of a flattened ellipsoids (Eq. 2.132 BT2)
    :param m: integrand variable
    :param R: Radius on the meridional plane
    :param e: flattening
    :return:
    """

    cdef:
        double m2=m*m
        double den

    den=sqrt(R*R - m2*e*e)

    return m2/den


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
    q=sqrt(1-e*e)
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
    
'''