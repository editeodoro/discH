#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin, exp, abs
import numpy as np

ctypedef double (*f_type)(double, double[:], int) nogil


cdef double poly_exponential(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:

    """

    :param R: variable
    :param a0: Radial scale length in Kpc
    :param a1: coeff-0 polynomial
    :param a2: coeff-1 polynomial
    :param a3: coeff-2 polynomial
    :param a4: coeff-3 polynomial
    :param a5: coeff-4 polynomial
    :param a6: coeff-5 polynomial
    :param a7: coeff-6 polynomial
    :param a8: coeff-7 polynomial
    :param a9: coeff-8 polynomial
    :return:
    """


    cdef:
        double res, recursiveR, Rd

    Rd=a0

    res=a1
    recursiveR=R
    res+=a2*recursiveR
    recursiveR=recursiveR*R
    res+=a3*recursiveR
    recursiveR=recursiveR*R
    res+=a4*recursiveR
    recursiveR=recursiveR*R
    res+=a5*recursiveR
    recursiveR=recursiveR*R
    res+=a6*recursiveR
    recursiveR=recursiveR*R
    res+=a7*recursiveR
    recursiveR=recursiveR*R
    res+=a8*recursiveR
    recursiveR=recursiveR*R
    res+=a9*recursiveR

    return res*exp(-R/Rd)

cpdef double poly_exponentialpy(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:

    """

    :param R: variable
    :param a0: Radial scale length in Kpc
    :param a1: coeff-0 polynomial
    :param a2: coeff-1 polynomial
    :param a3: coeff-2 polynomial
    :param a4: coeff-3 polynomial
    :param a5: coeff-4 polynomial
    :param a6: coeff-5 polynomial
    :param a7: coeff-6 polynomial
    :param a8: coeff-7 polynomial
    :param a9: coeff-8 polynomial
    :return:
    """


    cdef:
        double res, recursiveR, Rd

    Rd=a0

    res=a1
    recursiveR=R
    res+=a2*recursiveR
    recursiveR=recursiveR*R
    res+=a3*recursiveR
    recursiveR=recursiveR*R
    res+=a4*recursiveR
    recursiveR=recursiveR*R
    res+=a5*recursiveR
    recursiveR=recursiveR*R
    res+=a6*recursiveR
    recursiveR=recursiveR*R
    res+=a7*recursiveR
    recursiveR=recursiveR*R
    res+=a8*recursiveR
    recursiveR=recursiveR*R
    res+=a9*recursiveR

    return res*exp(-R/Rd)