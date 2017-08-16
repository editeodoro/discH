#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin, exp, abs, pow

ctypedef double (*f_type)(double, double[:], int) nogil

cdef double poly_flare(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:
    """
    Polynomial flaring
    :param R: Cilindrical radius
    :param param:
        Coefficent of the polynomial function (max 7h order), e.g. 3th order= param[0]+param[1]*x+param[2]*x*x+param[3]*x*x*x
        NB: a8 and a9 are special values used to make the scale heigth constant at a9 beyond a certain Radial limit a8
    :param ndim: length of the param list or order of the polynomial
    :return: Height scale at radius R
    """

    cdef:
        double res, recursiveR
        double Rlimit=a8
        double zdlimit=a9


    if (Rlimit>0) and (R>Rlimit):
        return zdlimit



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
    recursiveR=recursiveR*R
    res+=a6*recursiveR
    recursiveR=recursiveR*R
    res+=a7*recursiveR
    recursiveR=recursiveR*R


    return res


cdef double constant(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:
    """

    :param R:
    :param a0: constant zd
    :param a1:
    :param a2:
    :param a3:
    :param a4:
    :param a5:
    :param a6:
    :param a7:
    :param a8:
    :param a9:
    :return:
    """
    return a0

