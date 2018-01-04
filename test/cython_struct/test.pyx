ctypedef double (*f_type)(double, double[:], int)

cdef struct s_stu:

    f_type fu
    double a
    int i

ctypedef s_stu stu


