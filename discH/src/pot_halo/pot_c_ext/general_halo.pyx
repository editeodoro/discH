#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt

cdef double xi(double m,double R,double Z, double e) nogil:
    """Aux function for halo integration

    :param m:  elliptical radius
    :param R:  Cylindrical radius
    :param Z:  Cylindrical height
    :param e:  eccentricity
    :return:   xi function
    """

    return  (R*R+Z*Z+e*e*m*m+sqrt((e*m)**4-2*e*e*m*m*(R*R-Z*Z)+(R*R+Z*Z)*(R*R+Z*Z)))/(2*m*m)

cdef double m_calc(double R, double Z, double e) nogil:
    """Calculate the elliptical radius

    :param R: Cylindrical radius
    :param Z: Cylindrical height
    :param e: eccentricity
    :return:  elliptical radius
    """

    cdef:
        double q2=(1-e*e)

    return sqrt(R*R+Z*Z/q2)

