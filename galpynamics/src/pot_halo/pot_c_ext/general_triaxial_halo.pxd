#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False


cdef double integrand_core(double s, double a, double b, double c, double psi, double psicut) nogil
cdef double potential_core(double a, double b, double c, double intpot) nogil
#cdef double vcirc_core(double m, double R, double e) nogil

