#cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
from libc.math cimport sqrt, log, asin, exp, abs, pow, asinh, tanh, cosh, fabs

cdef double PI=3.14159265358979323846


#all normlised to the integral from -infty to infty.

cdef double zexp(double z, double zd) nogil:
    """
    Z=exp(-z/zd)/(2*zd)
    :param z:
    :param zd: scale height
    :return:
    """

    cdef:
        double norm, densz

    norm=(1/(2*zd))
    densz=exp(-fabs(z/zd))

    return norm*densz

cdef double zexp_der(double z, double zd) nogil:
    """
    Z=-exp(-z/zd)/(2*zd^2)=-zexp/zd
    :param z:
    :param zd: scale height
    :return:
    """
    cdef double zder=-zexp(z,zd)/(zd)

    return zder


cdef double zgau(double z, double zd) nogil:
    """
    Gau(z)=Exp(-0.5*z^2/zd^2) / (Sqrt(2*pi) *zd)
    :param z:
    :param zd: dispersion
    :return:
    """
    cdef:
        double norm, densz

    #3D dens
    norm=(1/(sqrt(2*PI)*zd))
    densz=exp(-0.5*(z/zd)*(z/zd))

    return densz*norm

cdef double zgau_der(double z, double zd) nogil:
    """
    Gau_der(z)=-Exp(-0.5*z^2/zd^2) / (Sqrt(2*pi) *zd) * (z/zd^2) = -Gau(z)*(z/zd^2)
    :param z:
    :param zd: dispersion
    :return:
    """
    cdef:
        double func, der_fact

    func=zgau(z, zd)
    der_fact=-z/(zd*zd)


    return func*der_fact

cdef double zsech2(double z, double zd) nogil:
    """
    Sech2(z)=(Sech(z/zd))^2 / ()
    :param z:
    :param zd: scale height
    :return:
    """

    cdef:
        double norm, densz

    norm=(1/(2*zd))
    densz=(1/(cosh(z/zd)) ) *  (1/(cosh(z/zd)) )

    return norm*densz

cdef double zsech2_der(double z, double zd) nogil:
    """
    Sech2_der(z)=-(Sech(z/zd))^2 / (2*zd) *  2*Tanh(z/zd)/zd=  - Sech2(z/zd) * 2*Tanh(z/zd)/zd
    :param z:
    :param zd: dispersion
    :return:
    """
    cdef:
        double func, der_fact

    func=zsech2(z, zd)
    der_fact=-2*tanh(z/zd)/zd


    return func*der_fact