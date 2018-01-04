from cython_gsl cimport *
import scipy.special as sf
from libc.math cimport sqrt, log, asin, exp, fabs, cosh
from scipy.special.cython_special cimport hyp2f1


def func(x):
    r = gsl_sf_ellint_Kcomp(x, GSL_PREC_DOUBLE)
    return r


def funcsc(x):
    r=sf.ellipk(x)
    return r

cdef double hyc(double a, double b, double c, double  x) nogil:

    return hyp2f1(a,b,c,x)

def hy(a,b,c,x):

    return hyc(a,b,c,x)
