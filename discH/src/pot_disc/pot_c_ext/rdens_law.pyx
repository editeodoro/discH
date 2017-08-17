#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin, exp, abs, pow
import numpy as np

ctypedef double (*f_type)(double, double[:], int) nogil

#All normalizes over sigma0

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


def poly_exponentialw(R, Rd, coeff):



    nlen=len(coeff)

    if nlen>0:
        res=0
        for i in range(nlen):
            res+=coeff[i]*R**i
        res=res/coeff[0]
    else:
        res=1


    return res*np.exp(-R/Rd)



cdef double gaussian(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:


    cdef:
        double sigmad=a0
        double R0=a1
        double earg, sigmad2, eargnorm,rexp, rexpnorm

    sigmad2=sigmad*sigmad

    earg=(R-R0)*(R-R0)/(sigmad2)
    eargnorm=(R0*R0)/(sigmad2)

    rexp=exp(-0.5*earg)
    rexpnorm=exp(-0.5*eargnorm)

    return rexp/rexpnorm

def gaussianw(R,sigmad,R0):

    sigmad2=sigmad*sigmad

    earg=(R-R0)*(R-R0)/(sigmad2)
    eargnorm=(R0*R0)/(sigmad2)

    rexp=np.exp(-0.5*earg)
    rexpnorm=np.exp(-0.5*eargnorm)

    return rexp/rexpnorm

cdef double fratlaw(double R, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8, double a9) nogil:

    cdef:
        double Rd=a0
        double Rd2=a1
        double alpha=a2
        double mult1, mult1a, mult2, xd, xd2


    xd=R/Rd
    xd2=R/Rd2
    mult1=(1+xd2)
    mult1a=pow(mult1,alpha)
    mult2=exp(-xd)

    return mult1a*mult2


def fratlaww(R, Rd, Rd2, alpha):

    xd=R/Rd
    xd2=R/Rd2
    mult1=(1+xd2)**alpha
    mult2=np.exp(-xd)

    return mult1*mult2




