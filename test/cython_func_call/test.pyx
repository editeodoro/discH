

ctypedef int (*f_type)(int) nogil


cdef int test(f_type f,int x) nogil:

    return f(x)


cdef int f1(int x) nogil:

    return x+1

cdef int f2(int x) nogil:

    return x+2


cpdef int fpy(int x, str option):

    if option=='1':

        return test(f1,x)

    if option=='2':

        return test(f2,x)

