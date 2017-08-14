#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin, exp, abs, pow

ctypedef double (*f_type)(double, double[:], int) nogil


cdef double poly_flare(double R, double a0, double a1, double a2, double a3, double a4, double a5) nogil:
    """
    Polynomial flaring
    :param R: Cilindrical radius
    :param param: Coefficent of the polynomial function, e.g. 3th order= param[0]+param[1]*x+param[2]*x*x+param[3]*x*x*x
    :param ndim: length of the param list or order of the polynomial
    :return: Height scale at radius R
    """

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

    return res

cpdef double poly_flarepy(double R, double a0, double a1, double a2, double a3, double a4, double a5):
    """
    Polynomial flaring
    :param R: Cilindrical radius
    :param param: Coefficent of the polynomial function, e.g. 3th order= param[0]+param[1]*x+param[2]*x*x+param[3]*x*x*x
    :param ndim: length of the param list or order of the polynomial
    :return: Height scale at radius R
    """

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

    return res


