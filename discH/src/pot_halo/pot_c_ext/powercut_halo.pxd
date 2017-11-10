#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False

cdef double dens_powercut(double m, void * params) nogil
cdef double psi_powercut(double d0, double rs, double rb, double alpha, double m, double toll) nogil
cdef double integrand_powercut(int nn, double *data) nogil
cdef double  _potential_powercut(double R, double Z, double mcut, double d0, double rs, double rb, double alpha, double e, double toll)
cdef double[:,:]  _potential_powercut_array(double[:] R, double[:] Z, int nlen, double mcut, double d0, double rs, double rb, double alpha, double e, double toll)
cdef double[:,:]  _potential_powercut_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, double mcut, double d0, double rs, double rb, double alpha, double e, double toll)
cdef double vcirc_integrand_powercut(int n, double *data) nogil
