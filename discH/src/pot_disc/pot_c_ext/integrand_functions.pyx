#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin, exp, fabs, cosh
from cython_gsl cimport *
from .rdens_law cimport poly_exponential
from .rflare_law cimport poly_flare
from scipy._lib._ccallback import LowLevelCallable
from scipy.integrate import nquad
cimport numpy as np
import numpy as np
import ctypes
cdef double PI=3.14159265358979323846



######
#ZEXP
cdef double zexp_dpoly_fpoly(double u, double l, double Rd, double d0, double d1, double d2, double d3, double d4, double d5, double f0, double f1, double f2, double f3, double f4, double f5 ) nogil:
    """Vertical law: Exp(-l/zd)/(2zd)
    Radial density: Polynomial + Exp
    Flare law: Polynomial
    l and zd need to have the same physical units
    :param u: integrand variable in R
    :param l: integrand variable in Z
    :param Rd: radial scale length
    :param d0-d5: polynomial coefficients for the surface density (d0=1,d1-d5=0 for a simple exponential)
    :param f0-f5: polynomial coefficients for the flare law (f0=zd, f1-f5=0 for a constant scale height)
    :return: The 3D density at R=u and Z=l
    """

    cdef:
        double zd, norm, densr, densz

    zd=poly_flare(u, f0, f1, f2,f3,f4,f5)
    norm=(1/(2*zd))
    densr=poly_exponential(u, Rd, d0, d1, d2, d3, d4, d5)
    densz=exp(-fabs(l/zd))

    #return 3.
    return densr*densz*norm

#Potential calc
cdef double integrand_zexp_dpoly_fpoly(int n, double *data) nogil:
    """Integrand function for
    Vertical law: Exp(-l/zd)/(2zd)
    Radial density: Polynomial + Exp
    Flare law: Polynomial
    l and zd need to have the same physical units

    :param data:
        0-u, Radial integration variable
        1-l, Vertical integration variable
        2-R, Radial position
        3-Z, Vertical position
        4-Rd, radial scale length
        5-10 rcoeff, polynomial coefficient of the surface density
        11-6 fcoeff, polynomial coefficients for the flare law
    :return: Value of the integrand function
    """

    n=17

    cdef:
        double u = data[0] #R intengration variable
        double l = data[1] #Z integration variable
        double R = data[2]
        double Z = data[3]


    if (R==0) | (u==0): return 0 #Singularity of the integral

    cdef:
        double Rd = data[4]
        double d0 = data[5]
        double d1 = data[6]
        double d2 = data[7]
        double d3 = data[8]
        double d4 = data[9]
        double d5 = data[10]
        double f0 = data[11]
        double f1 = data[12]
        double f2 = data[13]
        double f3 = data[14]
        double f4 = data[15]
        double f5 = data[16]
        double x, y,


    x=(R*R + u*u + (l-Z)*(l-Z))/(2*R*u)
    y=2/(x+1)
    dens=zexp_dpoly_fpoly(u,l,Rd,d0,d1,d2,d3,d4,d5,f0,f1,f2,f3,f4,f5)
    ellipkval=gsl_sf_ellint_Kcomp(sqrt(y), GSL_PREC_DOUBLE)

    if x==1.: return 0.
    else: return sqrt(u*y)*ellipkval*dens
##########

######
#ZGAU
cdef double zgau_dpoly_fpoly(double u, double l, double Rd, double d0, double d1, double d2, double d3, double d4, double d5, double f0, double f1, double f2, double f3, double f4, double f5 ) nogil:
    """Vertical law: Exp(-0.5*(l/zd)^2)/(sqrt(2 pi) zd)
    Radial density: Polynomial + Exp
    Flare law: Polynomial
    l and zd need to have the same physical units
    :param u: integrand variable in R
    :param l: integrand variable in Z
    :param Rd: radial scale length
    :param d0-d5: polynomial coefficients for the surface density (d0=1,d1-d5=0 for a simple exponential)
    :param f0-f5: polynomial coefficients for the flare law (f0=zd, f1-f5=0 for a constant scale height)
    :return: The 3D density at R=u and Z=l
    """

    cdef:
        double zd, norm, densr, densz


    zd=poly_flare(u, f0, f1, f2,f3,f4,f5)
    norm=(1/(sqrt(2*PI)*zd))
    densr=poly_exponential(u, Rd, d0, d1, d2, d3, d4, d5)
    densz=exp(-0.5*(l/zd)*(l/zd))

    return densr*densz*norm

#Potential calc
cdef double integrand_zgau_dpoly_fpoly(int n, double *data) nogil:
    """Integrand function for
    Vertical law: Exp(-0.5*(l/zd)^2)/(sqrt(2 pi) zd)
    Radial density: Polynomial + Exp
    Flare law: Polynomial
    l and zd need to have the same physical units

    :param data:
        0-u, Radial integration variable
        1-l, Vertical integration variable
        2-R, Radial position
        3-Z, Vertical position
        4-Rd, radial scale length
        5-10 rcoeff, polynomial coefficient of the surface density
        11-6 fcoeff, polynomial coefficients for the flare law
    :return: Value of the integrand function
    """

    n=17

    cdef:
        double u = data[0] #R intengration variable
        double l = data[1] #Z integration variable
        double R = data[2]
        double Z = data[3]


    if (R==0) | (u==0): return 0 #Singularity of the integral

    cdef:
        double Rd = data[4]
        double d0 = data[5]
        double d1 = data[6]
        double d2 = data[7]
        double d3 = data[8]
        double d4 = data[9]
        double d5 = data[10]
        double f0 = data[11]
        double f1 = data[12]
        double f2 = data[13]
        double f3 = data[14]
        double f4 = data[15]
        double f5 = data[16]
        double x, y,


    x=(R*R + u*u + (l-Z)*(l-Z))/(2*R*u)
    y=2/(x+1)
    dens=zgau_dpoly_fpoly(u,l,Rd,d0,d1,d2,d3,d4,d5,f0,f1,f2,f3,f4,f5)
    ellipkval=gsl_sf_ellint_Kcomp(sqrt(y), GSL_PREC_DOUBLE)

    if x==1.: return 0.
    else: return sqrt(u*y)*ellipkval*dens
##########


######
#ZSECH2
cdef double zsech2_dpoly_fpoly(double u, double l, double Rd, double d0, double d1, double d2, double d3, double d4, double d5, double f0, double f1, double f2, double f3, double f4, double f5 ) nogil:
    """Vertical law: Sech(-l/zd)^2
    Radial density: Polynomial + Exp
    Flare law: Polynomial
    l and zd need to have the same physical units
    :param u: integrand variable in R
    :param l: integrand variable in Z
    :param Rd: radial scale length
    :param d0-d5: polynomial coefficients for the surface density (d0=1,d1-d5=0 for a simple exponential)
    :param f0-f5: polynomial coefficients for the flare law (f0=zd, f1-f5=0 for a constant scale height)
    :return: The 3D density at R=u and Z=l
    """

    cdef:
        double zd, norm, densr, densz

    zd=poly_flare(u, f0, f1, f2,f3,f4,f5)
    norm=(1/(2*zd))
    densr=poly_exponential(u, Rd, d0, d1, d2, d3, d4, d5)
    densz=(1/(cosh(l/zd)) ) *  (1/(cosh(l/zd)) )

    return densr*densz*norm



#Potential calc
cdef double integrand_zsech2_dpoly_fpoly(int n, double *data) nogil:
    """Integrand function for
    Vertical law: Sech(-l/zd)^2
    Radial density: Polynomial + Exp
    Flare law: Polynomial
    l and zd need to have the same physical units

    :param data:
        0-u, Radial integration variable
        1-l, Vertical integration variable
        2-R, Radial position
        3-Z, Vertical position
        4-Rd, radial scale length
        5-10 rcoeff, polynomial coefficient of the surface density
        11-6 fcoeff, polynomial coefficients for the flare law
    :return: Value of the integrand function
    """

    n=17

    cdef:
        double u = data[0] #R intengration variable
        double l = data[1] #Z integration variable
        double R = data[2]
        double Z = data[3]


    if (R==0) | (u==0): return 0 #Singularity of the integral

    cdef:
        double Rd = data[4]
        double d0 = data[5]
        double d1 = data[6]
        double d2 = data[7]
        double d3 = data[8]
        double d4 = data[9]
        double d5 = data[10]
        double f0 = data[11]
        double f1 = data[12]
        double f2 = data[13]
        double f3 = data[14]
        double f4 = data[15]
        double f5 = data[16]
        double x, y,


    x=(R*R + u*u + (l-Z)*(l-Z))/(2*R*u)
    y=2/(x+1)
    dens=zsech2_dpoly_fpoly(u,l,Rd,d0,d1,d2,d3,d4,d5,f0,f1,f2,f3,f4,f5)
    ellipkval=gsl_sf_ellint_Kcomp(sqrt(y), GSL_PREC_DOUBLE)

    if x==1.: return 0.
    else: return sqrt(u*y)*ellipkval*dens
##########



#######
#Potential
cdef double _potential_disc_dpoly_fpoly(double R, double Z, int zlaw, double sigma0, double[:] rparam, double[:] fparam, double toll, double rcut, double zcut):
    """

    :param R:
    :param Z:
    :param zlaw:
    :param sigma0:
    :param rparam:
    :param fparam:
    :param toll:
    :param rcut:
    :param zcut:
    :return:
    """

    cdef:
        double G=4.498658966346282e-12 #kpc^2/(msol myr^2)
        double cost=-(2*G*sigma0)/(sqrt(R))
        double intpot



    #Integ
    import discH.src.pot_disc.pot_c_ext.integrand_functions as mod
    if zlaw==0:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zexp_dpoly_fpoly')
    elif zlaw==1:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zsech2_dpoly_fpoly')
    elif zlaw==2:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zgau_dpoly_fpoly')

    cdef:
        double Rd=rparam[0]
        double d0=rparam[1]
        double d1=rparam[2]
        double d2=rparam[3]
        double d3=rparam[4]
        double d4=rparam[5]
        double d5=rparam[6]
        double f0=fparam[0]
        double f1=fparam[1]
        double f2=fparam[2]
        double f3=fparam[3]
        double f4=fparam[4]
        double f5=fparam[5]

    intpot=nquad(fintegrand,[[0.,rcut],[-zcut,zcut]],args=(R,Z,Rd,d0,d1,d2,d3,d4,d5,f0,f1,f2,f3,f4,f5),opts=[({'points':[0,R],'epsabs':toll,'epsrel':toll}),({'points':[Z],'epsabs':toll,'epsrel':toll})])[0]

    return cost*intpot

#array
cdef double[:,:] _potential_disc_dpoly_fpoly_array(double[:] R, double[:] Z, int nlen , int zlaw, double sigma0, double[:] rparam, double[:] fparam, double toll, double rcut, double zcut):
    """

    :param R:
    :param Z:
    :param nlen:
    :param zlaw:
    :param sigma0:
    :param rparam:
    :param fparam:
    :param toll:
    :param rcut:
    :param zcut:
    :return:
    """

    cdef:
        double G=4.498658966346282e-12 #kpc^2/(msol myr^2)
        double cost=-(2*G*sigma0)
        double intpot
        double[:,:] ret=np.empty((nlen,3), dtype=np.dtype("d"))
        int i



    #Integ
    import discH.src.pot_disc.pot_c_ext.integrand_functions as mod
    if zlaw==0:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zexp_dpoly_fpoly')
    elif zlaw==1:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zsech2_dpoly_fpoly')
    elif zlaw==2:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zgau_dpoly_fpoly')

    cdef:
        double Rd=rparam[0]
        double d0=rparam[1]
        double d1=rparam[2]
        double d2=rparam[3]
        double d3=rparam[4]
        double d4=rparam[5]
        double d5=rparam[6]
        double f0=fparam[0]
        double f1=fparam[1]
        double f2=fparam[2]
        double f3=fparam[3]
        double f4=fparam[4]
        double f5=fparam[5]

    for  i in range(nlen):


        ret[i,0]=R[i]
        ret[i,1]=Z[i]


        intpot=nquad(fintegrand,[[0.,rcut],[-zcut,zcut]],args=(R[i],Z[i],Rd,d0,d1,d2,d3,d4,d5,f0,f1,f2,f3,f4,f5),opts=[({'points':[0,R],'epsabs':toll,'epsrel':toll}),({'points':[Z],'epsabs':toll,'epsrel':toll})])[0]


        ret[i,2]=(cost/(sqrt(R[i])))*intpot

    return ret


#grid
cdef double[:,:] _potential_disc_dpoly_fpoly_grid(double[:] R, double[:] Z, int nlenR, int nlenZ, int zlaw, double sigma0, double[:] rparam, double[:] fparam, double toll, double rcut, double zcut):
    """

    :param R:
    :param Z:
    :param nlenR:
    :param nlenZ:
    :param zlaw:
    :param sigma0:
    :param rparam:
    :param fparam:
    :param toll:
    :param rcut:
    :param zcut:
    :return:
    """

    cdef:
        double G=4.498658966346282e-12 #kpc^2/(msol myr^2)
        double cost=-(2*G*sigma0)
        double[:,:] ret=np.empty((nlenR*nlenZ,3), dtype=np.dtype("d"))
        double intpot
        int i, j, c



    #Integ
    import discH.src.pot_disc.pot_c_ext.integrand_functions as mod
    if zlaw==0:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zexp_dpoly_fpoly')
    elif zlaw==1:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zsech2_dpoly_fpoly')
    elif zlaw==2:
        fintegrand=LowLevelCallable.from_cython(mod,'integrand_zgau_dpoly_fpoly')

    cdef:
        double Rd=rparam[0]
        double d0=rparam[1]
        double d1=rparam[2]
        double d2=rparam[3]
        double d3=rparam[4]
        double d4=rparam[5]
        double d5=rparam[6]
        double f0=fparam[0]
        double f1=fparam[1]
        double f2=fparam[2]
        double f3=fparam[3]
        double f4=fparam[4]
        double f5=fparam[5]

    c=0
    for  i in range(nlenR):
        for j in range(nlenZ):

            ret[c,0]=R[i]
            ret[c,1]=Z[j]


            intpot=nquad(fintegrand,[[0.,rcut],[-zcut,zcut]],args=(R[i],Z[j],Rd,d0,d1,d2,d3,d4,d5,f0,f1,f2,f3,f4,f5),opts=[({'points':[0,R],'epsabs':toll,'epsrel':toll}),({'points':[Z],'epsabs':toll,'epsrel':toll})])[0]

            ret[c,2]=(cost/(sqrt(R[i])))*intpot

            c+=1

    return ret
####


cpdef potential_disc_dpoly_fpoly(R, Z, sigma0, Rd, rcoeff, fcoeff, zlaw='gau', rcut=None, zcut=None, toll=1e-4, grid=False):



    if zlaw=='exp': izdens=0
    elif zlaw=='sech2': izdens=1
    elif zlaw=='gau': izdens=2
    else: raise NotImplementedError()

    rparam=np.array([Rd,]+list(rcoeff),dtype=np.dtype("d"))
    fparam=np.array(fcoeff,dtype=np.dtype("d"))

    if rcut is None:
        rcut=2*Rd
    if zcut is None:
        zcut=rcut


    if isinstance(R, float) or isinstance(R, int):
        if isinstance(Z, float) or isinstance(Z, int):
            R=float(R)
            Z=float(Z)
            if rcut is None:
                rcut=2*R
            if zcut is None:
                zcut=2*Z

            ret=[R,Z,0]
            ret[2]=_potential_disc_dpoly_fpoly(R=R,Z=Z,zlaw=izdens,sigma0=sigma0,rparam=rparam,fparam=fparam,toll=toll,rcut=rcut,zcut=zcut)

            return np.array(ret)
        else:
            raise ValueError('R and Z have different dimension')
    else:
        if rcut is None:
            rcut=2*np.max(R)
        if zcut is None:
            zcut=2*np.max(np.abs(Z))

        if grid:
            return np.array(_potential_disc_dpoly_fpoly_grid(R=R,Z=Z,nlenR=len(R), nlenZ=len(Z),zlaw=izdens,sigma0=sigma0,rparam=rparam,fparam=fparam,toll=toll,rcut=rcut,zcut=zcut))
        elif len(R)==len(Z):
            nlen=len(R)
            R=np.array(R,dtype=np.dtype("d"))
            Z=np.array(Z,dtype=np.dtype("d"))
            return np.array(_potential_disc_dpoly_fpoly_array(R=R,Z=Z,nlen=len(R),zlaw=izdens,sigma0=sigma0,rparam=rparam,fparam=fparam,toll=toll,rcut=rcut,zcut=zcut))
        else:
            raise ValueError('R and Z have different dimension')

