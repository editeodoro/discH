from .pot_c_ext.integrand_functions import potential_disc, potential_disc_thin
import multiprocessing as mp
from ..pardo.Pardo import ParDo
import  numpy as np
from scipy.optimize import curve_fit

class disc:
    """
    Super class for halo potentials
    """
    def __init__(self,sigma0,rparam,fparam,zlaw,rlaw,flaw):
        """Init

        :param d0:  Central density in Msun/kpc^3
        :param rc:  Scale radius in kpc
        :param e:  eccentricity (sqrt(1-b^2/a^2))
        :param mcut: elliptical radius where dens(m>mcut)=0
        """

        self.sigma0=sigma0
        self.rparam=np.zeros(10)
        self.fparam=np.zeros(10)
        self.zlaw=zlaw
        self.rlaw=rlaw
        self.flaw=flaw
        self.lenparam = 10
        self.name ='General disc'

        #Make rparam
        lrparam=len(rparam)
        if lrparam>self.lenparam: raise ValueError('rparam length cannot exced %i'%self.lenparam)
        elif lrparam<self.lenparam: self.rparam[:lrparam]=rparam
        else: self.rparam[:]=rparam

        #Make fparam
        lfparam=len(fparam)
        if lfparam>self.lenparam: raise ValueError('fparam length cannot exced %i'%self.lenparam)
        elif lfparam<self.lenparam: self.fparam[:lfparam]=fparam
        else: self.fparam[:]=fparam

        if zlaw=='dirac':
            self._pot_serial = self._potential_serial_thin
            self._pot_parallel = self._potential_parallel_thin
        else:
            self._pot_serial = self._potential_serial
            self._pot_parallel =  self._potential_parallel



    def potential(self,R,Z,grid=False,toll=1e-4,Rcut=None, zcut=None, nproc=1):
        """Calculate potential at coordinate (R,Z). If R and Z are arrays with unequal lengths or
            if grid is True, the potential will be calculated in a 2D grid in R and Z.

        :param R: Cylindrical radius [kpc]
        :param Z: Cylindrical height [kpc]
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll: tollerance for quad integration
        :param mcut: elliptical radius where dens(m>mcut)=0
        :param nproc: Number of processes
        :return:  An array with:
            0-R
            1-Z
            2-Potential
        """

        if nproc==1:
            return self._pot_serial(R=R,Z=Z,grid=grid,toll=toll,Rcut=Rcut,zcut=zcut)
        else:
            if len(R)!=len(Z) or grid==True:
                ndim=len(R)*len(Z)
            else:
                ndim=len(R)

            return self._pot_parallel(R=R, Z=Z, grid=grid, toll=toll, Rcut=Rcut, zcut=zcut, nproc=nproc)

    def _potential_serial(self,R,Z,grid=False,toll=1e-4,Rcut=None, zcut=None, **kwargs):
        """Calculate the potential in R and Z using a serial code

        :param R: Cylindrical radius [kpc]
        :param Z: Cylindrical height [kpc]
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll: tollerance for quad integration
        :param mcut: elliptical radius where dens(m>mcut)=0
        :return:
        """


        if Rcut is not None: Rcut=Rcut
        elif (isinstance(R,float) or isinstance(R, int)): Rcut=3*R
        else: Rcut=3*np.max(R)

        if zcut is not None: zcut=zcut
        elif (isinstance(Z,float) or isinstance(Z, int)): zcut=10*Z
        else: zcut=10*np.max(Z)

        return potential_disc(R,Z,sigma0=self.sigma0, rcoeff=self.rparam, fcoeff=self.fparam,zlaw=self.zlaw, rlaw=self.rlaw, flaw=self.flaw,rcut=Rcut,zcut=zcut, toll=toll, grid=grid)


    def _potential_serial_thin(self,R,Z,grid=False,toll=1e-4,Rcut=None, **kwargs):
        """Calculate the potential in R and Z using a serial code

        :param R: Cylindrical radius [kpc]
        :param Z: Cylindrical height [kpc]
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll: tollerance for quad integration
        :param mcut: elliptical radius where dens(m>mcut)=0
        :return:
        """


        if Rcut is not None: Rcut=Rcut
        elif (isinstance(R,float) or isinstance(R, int)): Rcut=3*R
        else: Rcut=3*np.max(R)


        return potential_disc_thin(R,Z,sigma0=self.sigma0, rcoeff=self.rparam, rlaw=self.rlaw, rcut=Rcut, toll=toll, grid=grid)



    def _potential_parallel(self, R, Z, grid=False, toll=1e-4, Rcut=None, zcut=None, nproc=2, **kwargs):

        if Rcut is not None:
            Rcut = Rcut
        elif (isinstance(R, float) or isinstance(R, int)):
            Rcut = 3 * R
        else:
            Rcut = 3 * np.max(R)

        if zcut is not None:
            zcut = zcut
        elif (isinstance(Z, float) or isinstance(Z, int)):
            zcut = 10 * Z
        else:
            zcut = 10 * np.max(Z)


        pardo=ParDo(nproc=nproc)
        pardo.set_func(potential_disc)

        if len(R)!=len(Z) or grid==True:

            htab = pardo.run_grid(R,args=(Z,self.sigma0,self.rparam,self.fparam,self.zlaw,self.rlaw,self.flaw, Rcut, zcut, toll,grid))

        else:

            htab = pardo.run(R, Z, args=(self.sigma0, self.rparam, self.fparam, self.zlaw, self.rlaw, self.flaw, Rcut, zcut, toll, grid))


        return htab

    def _potential_parallel_thin(self, R, Z, grid=False, toll=1e-4, Rcut=None,  nproc=2, **kwargs):

        if Rcut is not None:
            Rcut = Rcut
        elif (isinstance(R, float) or isinstance(R, int)):
            Rcut = 3 * R
        else:
            Rcut = 3 * np.max(R)



        pardo=ParDo(nproc=nproc)
        pardo.set_func(potential_disc_thin)

        if len(R)!=len(Z) or grid==True:

            htab = pardo.run_grid(R,args=(Z,self.sigma0,self.rparam,self.rlaw, Rcut, toll, grid))

        else:

            htab = pardo.run(R, Z, args=(self.sigma0,self.rparam,self.rlaw, Rcut, toll, grid))


        return htab

    def __str__(self):

        s=''
        s+='Model: %s \n'%self.name
        s+='Sigma0: %.2f Msun/kpc2 \n'%self.sigma0
        s+='Vertical density law: %s\n'%self.zlaw
        s+='Radial density law: %s \n'%self.rlaw
        s+='Flaring law: %s \n'%self.flaw
        s+='Rparam: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n'%tuple(self.rparam)
        s+='Fparam: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n'%tuple(self.fparam)

        return s

class Exponential_disc(disc):

    def __init__(self,sigma0,Rd,fparam,zlaw='gau',flaw='poly'):

        rparam=np.array([Rd,1])
        self.Rd=Rd

        super(Exponential_disc,self).__init__(sigma0=sigma0,rparam=rparam,fparam=fparam,zlaw=zlaw,rlaw='epoly',flaw=flaw)
        self.name='Exponential disc'

    @classmethod
    def thin(cls,sigma0,Rd):

        fparam=np.array([0.0,0])


        return cls(sigma0=sigma0,Rd=Rd,fparam=fparam,zlaw='dirac',flaw='constant')

    @classmethod
    def thick(cls,sigma0, Rd, zd, zlaw='gau'):

        if zd<0.01:
            print('Warning Zd lower than 0.01, switching to thin disc')
            fparam=np.array([0,0])
            zlaw='dirac'
        else:
            fparam=np.array([zd,0])

        return cls(sigma0=sigma0, Rd=Rd, fparam=fparam, zlaw=zlaw, flaw='constant')

    @classmethod
    def polyflare(cls,sigma0,Rd, polycoeff, zlaw, Rlimit=None):

        lenp=len(polycoeff)
        if lenp>7:
            raise NotImplementedError('Polynomial flaring with order %i not implemented yet (max 7th)'%lenp)

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit=0
            for i in range(lenp):
                flimit+=polycoeff[i]*Rlimit**i

            #set fparam
            fparam=np.zeros(10)
            fparam[:lenp]=polycoeff
            fparam[-1]=flimit
            fparam[-2]=Rlimit

        else:
            fparam=polycoeff

        return cls(sigma0=sigma0, Rd=Rd, fparam=fparam, zlaw=zlaw, flaw='poly')

    @classmethod
    def asinhflare(cls,sigma0,Rd, h0, Rf, c, zlaw, Rlimit=None):

        fparam = np.zeros(10)
        fparam[0] = h0
        fparam[1] = Rf
        fparam[2] = c

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit=h0+c*np.arcsinh(Rlimit*Rlimit)
            fparam[-1] = flimit
            fparam[-2] = Rlimit

        return cls(sigma0=sigma0, Rd=Rd, fparam=fparam, zlaw=zlaw, flaw='asinh')

    @classmethod
    def tanhflare(cls, sigma0, Rd, h0, Rf, c, zlaw, Rlimit=None):

        fparam = np.zeros(10)
        fparam[0] = h0
        fparam[1] = Rf
        fparam[2] = c

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit = h0 + c * np.tanh(Rlimit * Rlimit)
            fparam[-1] = flimit
            fparam[-2] = Rlimit

        return cls(sigma0=sigma0, Rd=Rd, fparam=fparam, zlaw=zlaw, flaw='tanh')


    def change_flaring(self,flaw,zlaw=None,polycoeff=None,h0=None,c=None,Rf=None,zd=None,Rlimit=None):
        """
        Make a new object with the same radial surface density but different flaring
        :param flaw:
        :param polycoeff:
        :param h0:
        :param c:
        :param Rf:
        :param zd:
        :param Rlimit:
        :return:
        """

        sigma0=self.sigma0
        Rd=self.Rd
        if zlaw is None: zlaw=self.zlaw
        else: zlaw=zlaw


        if flaw=='thin':
            return Exponential_disc.thin(sigma0=sigma0,Rd=Rd)
        elif flaw=='thick':
            if zd is not None:
                return Exponential_disc.thick(sigma0=sigma0, Rd=Rd, zd=zd,zlaw=zlaw)
            else:
                raise ValueError('zd must be a non None value for thick flaw')
        elif flaw=='poly':
            if fcoeff is not None:
                return Exponential_disc.polyflare(sigma0=sigma0, Rd=Rd, polycoeff=polycoeff,zlaw=zlaw,Rlimit=Rlimit)
            else:
                raise ValueError('polycoeff must be a non None value for poly flaw')
        elif flaw=='asinh':
            if (h0 is not None) and (c is not None) and (Rf is not None):
                return Exponential_disc.asinhflare(sigma0=sigma0, Rd=Rd, h0=h0, c=c, Rf=Rf, zlaw=zlaw, Rlimit=Rlimit)
            else:
                raise ValueError('h0, c and Rf must be a non None values for asinh flaw')
        elif flaw=='asinh':
            if (h0 is not None) and (c is not None) and (Rf is not None):
                return Exponential_disc.tanhflare(sigma0=sigma0, Rd=Rd, h0=h0, c=c, Rf=Rf, zlaw=zlaw, Rlimit=Rlimit)
            else:
                raise ValueError('h0, c and Rf must be a non None values for tanh flaw')


    def __str__(self):

        s=''
        s+='Model: %s \n'%self.name
        s+='Sigma0: %.2f Msun/kpc2 \n'%self.sigma0
        s+='Vertical density law: %s\n'%self.zlaw
        s+='Radial density law: %s \n'%self.rlaw
        s+='Rd: %.3f kpc \n'%self.Rd
        s+='Flaring law: %s \n'%self.flaw
        s+='Fparam: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n'%tuple(self.fparam)

        return s


class PolyExponential_disc(disc):

    def __init__(self,sigma0,Rd,coeff,fparam,zlaw='gau',flaw='poly'):

        if isinstance(coeff,float) or isinstance(coeff,int):
            coeff=[1,]
        elif len(coeff):
            raise NotImplementedError('Maximum polynomial degree is 8')

        rparam=np.array([Rd,]+list(coeff))
        self.coeff=coeff
        self.Rd=Rd

        super(PolyExponential_disc,self).__init__(sigma0=sigma0,rparam=rparam,fparam=fparam,zlaw=zlaw,rlaw='epoly',flaw=flaw)
        self.name='PolyExponential disc'

    @classmethod
    def thin(cls,sigma0,Rd,coeff):

        fparam=np.array([0.0,0])


        return cls(sigma0=sigma0,Rd=Rd,coeff=coeff,fparam=fparam,zlaw='dirac',flaw='constant')

    @classmethod
    def thick(cls,sigma0, Rd, coeff, zd, zlaw='gau'):

        if zd<0.01:
            print('Warning Zd lower than 0.01, switching to thin disc')
            fparam=np.array([0,0])
            zlaw='dirac'
        else:
            fparam=np.array([zd,0])

        return cls(sigma0=sigma0, Rd=Rd, coeff=coeff, fparam=fparam, zlaw=zlaw, flaw='constant')

    @classmethod
    def polyflare(cls,sigma0,Rd, coeff, polycoeff, zlaw, Rlimit=None):

        lenp=len(polycoeff)
        if lenp>7:
            raise NotImplementedError('Polynomial flaring with order %i not implemented yet (max 7th)'%lenp)

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit=0
            for i in range(lenp):
                flimit+=polycoeff[i]*Rlimit**i

            #set fparam
            fparam=np.zeros(10)
            fparam[:lenp]=polycoeff
            fparam[-1]=flimit
            fparam[-2]=Rlimit

        else:
            fparam=polycoeff

        return cls(sigma0=sigma0, Rd=Rd, coeff=coeff, fparam=fparam, zlaw=zlaw, flaw='poly')


    @classmethod
    def asinhflare(cls,sigma0,Rd, coeff, h0, Rf, c, zlaw, Rlimit=None):

        fparam = np.zeros(10)
        fparam[0] = h0
        fparam[1] = Rf
        fparam[2] = c

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit=h0+c*np.arcsinh(Rlimit*Rlimit)
            fparam[-1] = flimit
            fparam[-2] = Rlimit

        return cls(sigma0=sigma0, Rd=Rd, coeff=coeff, fparam=fparam, zlaw=zlaw, flaw='asinh')

    @classmethod
    def tanhflare(cls, sigma0, Rd, coeff, h0, Rf, c, zlaw, Rlimit=None):

        fparam = np.zeros(10)
        fparam[0] = h0
        fparam[1] = Rf
        fparam[2] = c

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit = h0 + c * np.tanh(Rlimit * Rlimit)
            fparam[-1] = flimit
            fparam[-2] = Rlimit

        return cls(sigma0=sigma0, Rd=Rd, coeff=coeff, fparam=fparam, zlaw=zlaw, flaw='tanh')


    def change_flaring(self,flaw,zlaw=None,polycoeff=None,h0=None,c=None,Rf=None,zd=None,Rlimit=None):
        """
        Make a new object with the same radial surface density but different flaring
        :param flaw:
        :param polycoeff:
        :param h0:
        :param c:
        :param Rf:
        :param zd:
        :param Rlimit:
        :return:
        """

        sigma0=self.sigma0
        coeff=self.coeff
        Rd=self.Rd
        if zlaw is None: zlaw=self.zlaw
        else: zlaw=zlaw


        if flaw=='thin':
            return PolyExponential_disc.thin(sigma0=sigma0,Rd=Rd,coeff=coeff)
        elif flaw=='thick':
            if zd is not None:
                return PolyExponential_disc.thick(sigma0=sigma0, Rd=Rd, coeff=coeff, zd=zd,zlaw=zlaw)
            else:
                raise ValueError('zd must be a non None value for thick flaw')
        elif flaw=='poly':
            if fcoeff is not None:
                return PolyExponential_disc.polyflare(sigma0=sigma0, Rd=Rd, coeff=coeff, polycoeff=polycoeff,zlaw=zlaw,Rlimit=Rlimit)
            else:
                raise ValueError('polycoeff must be a non None value for poly flaw')
        elif flaw=='asinh':
            if (h0 is not None) and (c is not None) and (Rf is not None):
                return PolyExponential_disc.asinhflare(sigma0=sigma0, Rd=Rd, coeff=coeff, h0=h0, c=c, Rf=Rf, zlaw=zlaw, Rlimit=Rlimit)
            else:
                raise ValueError('h0, c and Rf must be a non None values for asinh flaw')
        elif flaw=='asinh':
            if (h0 is not None) and (c is not None) and (Rf is not None):
                return PolyExponential_disc.tanhflare(sigma0=sigma0, Rd=Rd, coeff=coeff, h0=h0, c=c, Rf=Rf, zlaw=zlaw, Rlimit=Rlimit)
            else:
                raise ValueError('h0, c and Rf must be a non None values for tanh flaw')

    def __str__(self):

        s=''
        s+='Model: %s \n'%self.name
        s+='Sigma0: %.2f Msun/kpc2 \n'%self.sigma0
        s+='Vertical density law: %s\n'%self.zlaw
        s+='Radial density law: %s \n'%self.rlaw
        s+='Rd: %.3f kpc \n'%self.Rd
        s+='Polycoeff: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n'%tuple(self.coeff)
        s+='Flaring law: %s \n'%self.flaw
        s+='Fparam: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n'%tuple(self.fparam)

        return s

class Gaussian_disc(disc):

    def __init__(self,sigma0, sigmad, R0, fparam,zlaw='gau',flaw='poly'):

        rparam    = np.zeros(10)
        rparam[0] = sigmad
        rparam[1] = R0
        self.sigmad=sigmad
        self.R0=R0

        super(Gaussian_disc,self).__init__(sigma0=sigma0,rparam=rparam,fparam=fparam,zlaw=zlaw,rlaw='gau',flaw=flaw)
        self.name='Gaussian disc'

    @classmethod
    def thin(cls,sigma0,sigmad,R0):

        fparam=np.array([0.0,0])


        return cls(sigma0=sigma0,sigmad=sigmad,R0=R0,fparam=fparam,zlaw='dirac',flaw='constant')

    @classmethod
    def thick(cls,sigma0, sigmad, R0, zd, zlaw='gau'):

        if zd<0.01:
            print('Warning Zd lower than 0.01, switching to thin disc')
            fparam=np.array([0,0])
            zlaw='dirac'
        else:
            fparam=np.array([zd,0])

        return cls(sigma0=sigma0, sigmad=sigmad, R0=R0, fparam=fparam, zlaw=zlaw, flaw='constant')

    @classmethod
    def polyflare(cls,sigma0, sigmad, R0, polycoeff, zlaw, Rlimit=None):

        lenp=len(polycoeff)
        if lenp>7:
            raise NotImplementedError('Polynomial flaring with order %i not implemented yet (max 7th)'%lenp)

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit=0
            for i in range(lenp):
                flimit+=polycoeff[i]*Rlimit**i

            #set fparam
            fparam=np.zeros(10)
            fparam[:lenp]=polycoeff
            fparam[-1]=flimit
            fparam[-2]=Rlimit

        else:
            fparam=polycoeff

        return cls(sigma0=sigma0, sigmad=sigmad, R0=R0, fparam=fparam, zlaw=zlaw, flaw='poly')


    @classmethod
    def asinhflare(cls,sigma0, sigmad, R0, h0, Rf, c, zlaw, Rlimit=None):

        fparam = np.zeros(10)
        fparam[0] = h0
        fparam[1] = Rf
        fparam[2] = c

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit=h0+c*np.arcsinh(Rlimit*Rlimit)
            fparam[-1] = flimit
            fparam[-2] = Rlimit

        return cls(sigma0=sigma0, sigmad=sigmad, R0=R0, fparam=fparam, zlaw=zlaw, flaw='asinh')

    @classmethod
    def tanhflare(cls, sigma0, sigmad, R0, h0, Rf, c, zlaw, Rlimit=None):

        fparam = np.zeros(10)
        fparam[0] = h0
        fparam[1] = Rf
        fparam[2] = c

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit = h0 + c * np.tanh(Rlimit * Rlimit)
            fparam[-1] = flimit
            fparam[-2] = Rlimit

        return cls(sigma0=sigma0, sigmad=sigmad, R0=R0, fparam=fparam, zlaw=zlaw, flaw='tanh')

    def change_flaring(self,flaw,zlaw=None,polycoeff=None,h0=None,c=None,Rf=None,zd=None,Rlimit=None):
        """
        Make a new object with the same radial surface density but different flaring
        :param flaw:
        :param polycoeff:
        :param h0:
        :param c:
        :param Rf:
        :param zd:
        :param Rlimit:
        :return:
        """

        sigma0=self.sigma0
        sigmad=self.sigmad
        R0=self.R0
        if zlaw is None: zlaw=self.zlaw
        else: zlaw=zlaw


        if flaw=='thin':
            return Gaussian_disc.thin(sigma0=sigma0,sigmad=sigmad,R0=R0)
        elif flaw=='thick':
            if zd is not None:
                return Gaussian_disc.thick(sigma0=sigma0, sigmad=sigmad,R0=R0, zd=zd,zlaw=zlaw)
            else:
                raise ValueError('zd must be a non None value for thick flaw')
        elif flaw=='poly':
            if fcoeff is not None:
                return Gaussian_disc.polyflare(sigma0=sigma0, sigmad=sigmad,R0=R0, polycoeff=polycoeff,zlaw=zlaw,Rlimit=Rlimit)
            else:
                raise ValueError('polycoeff must be a non None value for poly flaw')
        elif flaw=='asinh':
            if (h0 is not None) and (c is not None) and (Rf is not None):
                return Gaussian_disc.asinhflare(sigma0=sigma0, sigmad=sigmad,R0=R0, h0=h0, c=c, Rf=Rf, zlaw=zlaw, Rlimit=Rlimit)
            else:
                raise ValueError('h0, c and Rf must be a non None values for asinh flaw')
        elif flaw=='asinh':
            if (h0 is not None) and (c is not None) and (Rf is not None):
                return Gaussian_disc.tanhflare(sigma0=sigma0, sigmad=sigmad,R0=R0, h0=h0, c=c, Rf=Rf, zlaw=zlaw, Rlimit=Rlimit)
            else:
                raise ValueError('h0, c and Rf must be a non None values for tanh flaw')

    def __str__(self):

        s=''
        s+='Model: %s \n'%self.name
        s+='Sigma0: %.2f Msun/kpc2 \n'%self.sigma0
        s+='Vertical density law: %s\n'%self.zlaw
        s+='Radial density law: %s \n'%self.rlaw
        s+='sigmad: %.3f kpc \n'%self.sigmad
        s+='R0: %.3f kpc \n'%self.R0
        s+='Flaring law: %s \n'%self.flaw
        s+='Fparam: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n'%tuple(self.fparam)

        return s


class Frat_disc(disc):

    def __init__(self,sigma0,Rd, Rd2, alpha, fparam, zlaw='gau', flaw='poly'):


        self.sigma0=sigma0
        self.Rd=Rd
        self.Rd2=Rd2
        self.alpha=alpha
        self.zlaw=zlaw

        rparam=np.zeros(10)
        rparam[0] = Rd
        rparam[1] = Rd2
        rparam[2] = alpha

        super(Frat_disc,self).__init__(sigma0=sigma0,rparam=rparam,fparam=fparam,zlaw=zlaw,rlaw='fratlaw',flaw=flaw)
        self.name='Frat disc'

    @classmethod
    def thin(cls, sigma0=None, Rd=None, Rd2=None, alpha=None, rfit_array=None):

        if rfit_array is not None:

            func_fit=lambda R, s0, Rd, Rd2, alpha: s0*np.exp(-R/Rd)*(1+(R/Rd2))**(alpha)
            if rfit_array.shape[1]==2:
                R=rfit_array[:,0]
                Sigma=rfit_array[:,1]
                Sigma_err=None
            elif rfit_array.shape[1]==3:
                R=rfit_array[:,0]
                Sigma=rfit_array[:,1]
                Sigma_err=rfit_array[:,2]
            else:
                raise ValueError('Wrong rfit dimension')

            p0=(Sigma[0],np.mean(R),np.mean(R),1)
            popt,pcov=curve_fit(f=func_fit,xdata=R,ydata=Sigma,sigma=Sigma_err,absolute_sigma=True,p0=p0)

            sigma0,Rd,Rd2,alpha=popt



        elif (sigma0 is not None) and (Rd is not None) and (Rd2 is not None) and (alpha is not None):
            pass
        else:
            raise ValueError()

        fparam = np.array([0.0, 0])

        return cls(sigma0=sigma0,Rd=Rd, alpha=alpha, Rd2=Rd2, fparam=fparam,zlaw='dirac',flaw='constant')


    @classmethod
    def thick(cls,sigma0, Rd, Rd2, alpha, zd, zlaw='gau'):

        if zd<0.01:
            print('Warning Zd lower than 0.01, switching to thin disc')
            fparam=np.array([0,0])
            zlaw='dirac'
        else:
            fparam=np.array([zd,0])

        return cls(sigma0=sigma0, Rd=Rd, alpha=alpha, Rd2=Rd2, fparam=fparam, zlaw=zlaw, flaw='constant')

    @classmethod
    def polyflare(cls,sigma0,Rd, Rd2, alpha, polycoeff, zlaw, Rlimit=None):

        lenp=len(polycoeff)
        if lenp>7:
            raise NotImplementedError('Polynomial flaring with order %i not implemented yet (max 7th)'%lenp)

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit=0
            for i in range(lenp):
                flimit+=polycoeff[i]*Rlimit**i

            #set fparam
            fparam=np.zeros(10)
            fparam[:lenp]=polycoeff
            fparam[-1]=flimit
            fparam[-2]=Rlimit

        else:
            fparam=polycoeff

        return cls(sigma0=sigma0, Rd=Rd, Rd2=Rd2, alpha=alpha, fparam=fparam, zlaw=zlaw, flaw='poly')


    @classmethod
    def asinhflare(cls, sigma0, Rd, Rd2, alpha, h0, Rf, c, zlaw, Rlimit=None):
        fparam = np.zeros(10)
        fparam[0] = h0
        fparam[1] = Rf
        fparam[2] = c

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit = h0 + c * np.arcsinh(Rlimit * Rlimit)
            fparam[-1] = flimit
            fparam[-2] = Rlimit

        return cls(sigma0=sigma0, Rd=Rd, Rd2=Rd2, alpha=alpha, fparam=fparam, zlaw=zlaw, flaw='asinh')



    @classmethod
    def tanhflare(cls, sigma0, Rd, Rd2, alpha, h0, Rf, c, zlaw, Rlimit=None):
        fparam = np.zeros(10)
        fparam[0] = h0
        fparam[1] = Rf
        fparam[2] = c

        if Rlimit is not None:
            # Calculate the value of Zd at Rlim
            flimit = h0 + c * np.tanh(Rlimit * Rlimit)
            fparam[-1] = flimit
            fparam[-2] = Rlimit

        return cls(sigma0=sigma0, Rd=Rd, Rd2=Rd2, alpha=alpha, fparam=fparam, zlaw=zlaw, flaw='tanh')


    def change_flaring(self,flaw,zlaw=None,polycoeff=None,h0=None,c=None,Rf=None,zd=None,Rlimit=None):
        """
        Make a new object with the same radial surface density but different flaring
        :param flaw:
        :param polycoeff:
        :param h0:
        :param c:
        :param Rf:
        :param zd:
        :param Rlimit:
        :return:
        """

        sigma0=self.sigma0
        Rd=self.Rd
        Rd2=self.Rd2
        alpha=self.alpha
        if zlaw is None: zlaw=self.zlaw
        else: zlaw=zlaw


        if flaw=='thin':
            return Frat_disc.thin(sigma0=sigma0,Rd=Rd,Rd2=Rd2,alpha=alpha)
        elif flaw=='thick':
            if zd is not None:
                return Frat_disc.thick(sigma0=sigma0, Rd=Rd, Rd2=Rd2, alpha=alpha, zd=zd,zlaw=zlaw)
            else:
                raise ValueError('zd must be a non None value for thick flaw')
        elif flaw=='poly':
            if fcoeff is not None:
                return Frat_disc.polyflare(sigma0=sigma0, Rd=Rd, Rd2=Rd2, alpha=alpha, polycoeff=polycoeff,zlaw=zlaw,Rlimit=Rlimit)
            else:
                raise ValueError('polycoeff must be a non None value for poly flaw')
        elif flaw=='asinh':
            if (h0 is not None) and (c is not None) and (Rf is not None):
                return Frat_disc.asinhflare(sigma0=sigma0, Rd=Rd, Rd2=Rd2, alpha=alpha, h0=h0, c=c, Rf=Rf, zlaw=zlaw, Rlimit=Rlimit)
            else:
                raise ValueError('h0, c and Rf must be a non None values for asinh flaw')
        elif flaw=='asinh':
            if (h0 is not None) and (c is not None) and (Rf is not None):
                return Frat_disc.tanhflare(sigma0=sigma0, Rd=Rd, Rd2=Rd2, alpha=alpha, h0=h0, c=c, Rf=Rf, zlaw=zlaw, Rlimit=Rlimit)
            else:
                raise ValueError('h0, c and Rf must be a non None values for tanh flaw')


    def __str__(self):

        s=''
        s+='Model: %s \n'%self.name
        s+='Sigma0: %.2f Msun/kpc2 \n'%self.sigma0
        s+='Vertical density law: %s\n'%self.zlaw
        s+='Radial density law: %s \n'%self.rlaw
        s+='Rd: %.3f kpc \n'%self.Rd
        s+='Rd2: %.3f kpc \n'%self.Rd2
        s+='alpha: %.1f\n'%self.alpha
        s+='Flaring law: %s \n'%self.flaw
        s+='Fparam: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n'%tuple(self.fparam)

        return s