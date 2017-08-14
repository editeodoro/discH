#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False

ctypedef double (*f_type)(double, double[:], int) nogil
cdef double zexp_dpoly_fpoly(double u, double l, double Rd, double d0, double d1, double d2, double d3, double d4, double d5, double f0, double f1, double f2, double f3, double f4, double f5 ) nogil
cdef double integrand_zexp_dpoly_fpoly(int n, double *data) nogil
cdef double integrand_zsech2_dpoly_fpoly(int n, double *data) nogil
cdef double integrand_zgau_dpoly_fpoly(int n, double *data) nogil




