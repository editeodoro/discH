#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False

cdef double psi_triaxial_exponential(double d0, double rs, double alpha, double m) nogil
cdef double integrand_triaxial_exponential(int nn, double *data) nogil
cdef double  _potential_triaxial_exponential(double x, double y, double z, double mcut, double d0, double rs, double alpha, double a, double b, double c, double toll)
cdef double[:,:]  _potential_triaxial_exponential_array(double[:] x, double[:] y, double[:] z, int nlen, double mcut, double d0, double rs, double alpha, double a, double b, double c, double toll)
cdef double[:,:]  _potential_triaxial_exponential_grid(double[:] x, double[:] y, double[:] z, int nlenx, int nleny, int nlenz, double mcut, double d0, double rs, double alpha, double a, double b, double c, double toll)
#cdef double vcirc_integrand_triaxial_exponential(int n, double *data) nogil
 