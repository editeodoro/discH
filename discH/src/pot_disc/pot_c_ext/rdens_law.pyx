#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin, exp, abs
import numpy as np

ctypedef double (*f_type)(double, double[:], int) nogil


cdef double exponential_disc(double R, double Rd, int nparam) nogil:
    """
    Normalised Exponetial disc surface density Sigma(R)=exp(-R/Rd)
    :param R: Cylindircal radius
    :param param:  array with 0-Sigma0 in Msun/kpc2 1-Rd in kpc
    :param nparam: not used
    :return: surface density at R
    """

    nparam=1



    return exp(-R/Rd)


cdef double poly_exponential(double R, double Rd, double a0, double a1, double a2, double a3, double a4, double a5) nogil:

    cdef:
        double res, recursiveR

    res=a0
    recursiveR=R
    res+=a1*recursiveR
    recursiveR=recursiveR*R
    res+=a2*recursiveR
    recursiveR=recursiveR*R
    res+=a3*recursiveR
    recursiveR=recursiveR*R
    res+=a4*recursiveR
    recursiveR=recursiveR*R
    res+=a5*recursiveR

    return res*exp(-R/Rd)

cpdef double poly_exponentialpy(double R, double Rd, double a0, double a1, double a2, double a3, double a4, double a5):

    cdef:
        double res, recursiveR

    res=a0
    recursiveR=R
    res+=a1*recursiveR
    recursiveR=recursiveR*R
    res+=a2*recursiveR
    recursiveR=recursiveR*R
    res+=a3*recursiveR
    recursiveR=recursiveR*R
    res+=a4*recursiveR
    recursiveR=recursiveR*R
    res+=a5*recursiveR

    return res*exp(-R/Rd)
