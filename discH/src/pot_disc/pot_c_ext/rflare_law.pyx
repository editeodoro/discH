#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin, exp, abs, pow, asinh, tanh, cosh
import numpy as np

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

cdef double poly_flare_der(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:
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
        double Rlimit=a8
        double der_zero=0. #When the zd is constant (R>rlimit) the derivative is 0.
        double der

    der=poly_flare(R, a1, 2*a2, 3*a3, 4*a4, 5*a5, 6*a6, 7*a7, 0, Rlimit, der_zero)

    return der

def poly_flarew(R, coeff, Rlimit=None):
    """
    Polynomial flaring
    :param R: Cilindrical radius
    :param param:
        Coefficent of the polynomial function (max 7h order), e.g. 3th order= param[0]+param[1]*x+param[2]*x*x+param[3]*x*x*x
        NB: a8 and a9 are special values used to make the scale heigth constant at a9 beyond a certain Radial limit a8
    :param ndim: length of the param list or order of the polynomial
    :return: Height scale at radius R
    """

    nlen=len(coeff)

    if Rlimit is not None:

        zdlimit=0
        for i in range(nlen):
            zdlimit+=coeff[i]*Rlimit**i

    else:
        zdlimit=0
        Rlimit=np.max(R)+1

    res=0
    for i in range(nlen):
        res+=coeff[i]*R**i

    return np.where(R>Rlimit,zdlimit,res)

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

cdef double constant_der(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:
    """

    :param R:
    :param a0:
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
    return 0

cpdef double constantw(double R, double zd):
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
    return zd

cdef double asinh_flare(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:
    """

    :param R:
    :param a0: h0, i.e., zd at R=0
    :param a1: Rf
    :param a2: c
    :param a3:
    :param a4:
    :param a5:
    :param a6:
    :param a7:
    :param a8: Rlimit
    :param a9: zdlimit
    :return:
    """

    cdef:
        double h0=a0
        double Rf=a1
        double c=a2
        double Rlimit=a8
        double zdlimit=a9
        double x

    if (Rlimit>0) and (R>Rlimit):
        return zdlimit

    x=R/Rf

    return  h0+c*asinh(x*x)

cdef double asinh_flare_der(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:
    """
    der= (2*c*x)/(sqrt(1+x^4)*Rf) dove x=R/Rf
    :param R:
    :param a0:
    :param a1: Rf
    :param a2: c
    :param a3:
    :param a4:
    :param a5:
    :param a6:
    :param a7:
    :param a8: Rlimit
    :param a9:
    :return:
    """

    cdef:
        double h0=a0
        double Rf=a1
        double c=a2
        double Rlimit=a8
        double der_zero=0.
        double x, num, den

    if (Rlimit>0) and (R>Rlimit): #where zd is constant derivative is 0
        return der_zero

    x=R/Rf
    num=2*c*x
    den=sqrt(1+x*x*x*x)*Rf

    return  num/den


def asinh_flarew(R, h0, Rf, c, Rlimit=None):
    """

    :param R:
    :param a0:
    :param a1: Rf
    :param a2: c
    :param a3:
    :param a4:
    :param a5:
    :param a6:
    :param a7:
    :param a8: Rlimit
    :param a9:
    :return:
    """


    if Rlimit is not None:
        xl=Rlimit/Rf
        zdlimit=h0+c*np.arcsinh(xl*xl)
    else:
        zdlimit=0
        Rlimit=np.max(R)+1

    x=R/Rf
    y=h0+c*np.arcsinh(x*x)

    return  np.where(R>Rlimit,zdlimit,y)

cdef double tanh_flare(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:
    """

    :param R:
    :param a0: h0, i.e., zd at R=0
    :param a1: Rf
    :param a2: c
    :param a3:
    :param a4:
    :param a5:
    :param a6:
    :param a7:
    :param a8: Rlimit
    :param a9: zdlimit
    :return:
    """

    cdef:
        double h0=a0
        double Rf=a1
        double c=a2
        double Rlimit=a8
        double zdlimit=a9
        double x

    if (Rlimit>0) and (R>Rlimit):
        return zdlimit

    x=R/Rf

    return  h0+c*tanh(x*x)

cdef double tanh_flare_der(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:
    """
    der 2*c*x*(Sech(x^2))^2 / Rf

    :param R:
    :param a0:
    :param a1: Rf
    :param a2: c
    :param a3:
    :param a4:
    :param a5:
    :param a6:
    :param a7:
    :param a8: Rlimit
    :param a9:
    :return:
    """

    cdef:
        double h0=a0
        double Rf=a1
        double c=a2
        double Rlimit=a8
        double der_zero=0.
        double x, num, den,sech


    if (Rlimit>0) and (R>Rlimit): #where zd is constant derivative is 0
        return der_zero

    x=R/Rf
    sech=(1/cosh(x*x))
    num=2*c**sech*sech
    den=Rf

    return  num/den

def tanh_flarew(R, h0, Rf, c, Rlimit=None):
    """

    :param R:
    :param a0: h0, i.e., zd at R=0
    :param a1: Rf
    :param a2: c
    :param a3:
    :param a4:
    :param a5:
    :param a6:
    :param a7:
    :param a8: Rlimit
    :param a9: zdlimit
    :return:
    """


    if Rlimit is not None:
        xl=Rlimit/Rf
        zdlimit=h0+c*np.tanh(xl*xl)
    else:
        zdlimit=0
        Rlimit=np.max(R)+1

    x=R/Rf
    y=h0+c*np.tanh(x*x)

    return  np.where(R>Rlimit,zdlimit,y)
