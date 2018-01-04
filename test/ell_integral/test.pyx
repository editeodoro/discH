from cython_gsl cimport *
import scipy.special as sf
from libc.math cimport sqrt, log, asin, exp, fabs, cosh
from scipy.special.cython_special cimport ellipe, ellipkinc, ellipkinc

cdef PI=3.14159265358979323846


def ellipe_sc(p):

    return ellipe(p)

def ellipk_sc(p):

    return ellipkinc(PI/2.,p)

def ellipe_gs(p):

    return gsl_sf_ellint_Ecomp(p, GSL_PREC_DOUBLE)


def ellipk_gs(p):

    return gsl_sf_ellint_Kcomp(p, GSL_PREC_DOUBLE)

