from .pot_c_ext.integrand_functions import potential_disc, potential_disc_thin
import multiprocessing as mp
from ..pardo.Pardo import ParDo
import  numpy as np

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


class Exponential_disc(disc):

    def __init__(self,sigma0,Rd,fparam,zlaw='gau',flaw='poly'):

        rparam=np.array([Rd,1])


        super(Exponential_disc,self).__init__(sigma0=sigma0,rparam=rparam,fparam=fparam,zlaw=zlaw,rlaw='epoly',flaw=flaw)

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



'''
class Poly_disc(disc):

    def __init__(self):


class Frat_disc(disc):

    def __init__(self):
'''

