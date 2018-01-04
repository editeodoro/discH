from cython_gsl cimport *
import numpy as np
cimport numpy as np
from scipy.interpolate import dfitpack
from scipy.interpolate import UnivariateSpline

cdef double[:,:] interp(double[:] x, double[:] xa, double[:] ya, int lenga, int lengx):

    cdef:
        gsl_interp_accel *acc
        gsl_spline *spline
        double[:,:] ret=np.empty((lengx,3), dtype=np.dtype("d"))
        int i

    acc= gsl_interp_accel_alloc ()
    spline = gsl_spline_alloc (gsl_interp_cspline, lenga)

    gsl_spline_init (spline, &xa[0], &ya[0], lenga)


    for i in range(lengx):
        xi=x[i]
        ret[i,0]=xi
        if (xi>=xa[0]) and (xi<=xa[-1]):
            yi = gsl_spline_eval (spline, xi, acc)
        elif xi<xa[0]:
            yi = gsl_spline_eval (spline, xa[0], acc)
        elif xi>xa[-1]:
            yi = gsl_spline_eval (spline, xa[-1], acc)
        ret[i,1]=yi

    gsl_spline_free (spline)
    gsl_interp_accel_free (acc)

    return ret

def interpy(x,xa,ya):

    x=np.array(x,dtype=np.dtype("d"))
    xa=np.array(xa,dtype=np.dtype("d"))
    ya=np.array(ya,dtype=np.dtype("d"))
    lengx=len(x)
    lenga=len(xa)

    retarr=interp(x,xa,ya,lenga,lengx)

    return retarr

cdef double interpu(double[:] x, double[:] y, int k, double s) nogil:

    cdef:
        double[:] w, t, c, fpint
        int[:] nrdata
        double xe,xb,fp
        int  n, ier


    x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier = dfitpack.fpcurf0(x,y,k,w=None,xb=None,xe=None,s=s)

    return 1.

def interpupy(x,y,k=2,s=0.):
    x=np.array(x,dtype=np.dtype("d"))
    y=np.array(y,dtype=np.dtype("d"))

    return interpu(x,y,k,s)



