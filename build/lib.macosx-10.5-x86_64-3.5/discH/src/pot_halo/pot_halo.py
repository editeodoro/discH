from .pot_c_ext.isothermal_halo import potential_iso
import multiprocessing as mp
from ..pardo.Pardo import ParDo

class halo:
    """
    Super class for halo potentials
    """
    def __init__(self,d0,rc,e,mcut=None):
        """Init

        :param d0:  Central density in Msun/kpc^3
        :param rc:  Scale radius in kpc
        :param e:  eccentricity (sqrt(1-b^2/a^2))
        :param mcut: elliptical radius where dens(m>mcut)=0
        """

        self.d0=d0
        self.rc=rc
        self.e=e
        self.toll=1e-4
        if mcut is None:
            self.mcut=20*self.rc
        else:
            self.mcut=mcut

    def set_toll(self,toll):
        """Set tollerance for quad integration

        :param toll: tollerance for quad integration
        :return:
        """

        self.toll=toll

    def set_mcut(self,mcut):
        """Set mcut

        :param mcut: elliptical radius where dens(m>mcut)=0
        :return:
        """

        self.toll=mcut

    def potential(self,R,Z,grid=False,toll=1e-4,mcut=None, nproc=1):
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
            return self._potential_serial(R=R,Z=Z,grid=grid,toll=toll,mcut=mcut)
        else:
            if len(R)!=len(Z) or grid==True:
                ndim=len(R)*len(Z)
            else:
                ndim=len(R)

            if ndim<100000: print('WARNING, too few point to exploit parallelization')

            return self._potential_parallel(R=R, Z=Z, grid=grid, toll=toll, mcut=mcut,nproc=nproc)

    def _potential_serial(self,R,Z,grid=False,toll=1e-4,mcut=None):
        """Calculate the potential in R and Z using a serial code

        :param R: Cylindrical radius [kpc]
        :param Z: Cylindrical height [kpc]
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll: tollerance for quad integration
        :param mcut: elliptical radius where dens(m>mcut)=0
        :return:
        """

        raise NotImplementedError('Potential serial not implemented for this class')

    def _potential_parallel(self, R, Z, grid=False, toll=1e-4, mcut=None,nproc=2):
        """Calculate the potential in R and Z using a parallelized code.

        :param R: Cylindrical radius [kpc]
        :param Z: Cylindrical height [kpc]
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll: tollerance for quad integration
        :param mcut: elliptical radius where dens(m>mcut)=0
        :return:
        """
        raise NotImplementedError('Potential serial not implemented for this class')

class isothermal_halo(halo):

    def __init__(self,d0,rc,e,mcut=None):
        """Isothermal halo d=d0/(1+r^2/rc^2)

        :param d0:  Central density in Msun/kpc^3
        :param rc:  Scale radius in kpc
        :param e:  eccentricity (sqrt(1-b^2/a^2))
        :param mcut: elliptical radius where dens(m>mcut)=0
        """
        super(isothermal_halo,self).__init__(d0=d0,rc=rc,e=e,mcut=mcut)

    def _potential_serial(self, R, Z, grid=False, toll=1e-4, mcut=None):
        """Calculate the potential in R and Z using a serial code

        :param R: Cylindrical radius [kpc]
        :param Z: Cylindrical height [kpc]
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll: tollerance for quad integration
        :param mcut: elliptical radius where dens(m>mcut)=0
        :return:
        """


        self.set_toll(toll)

        if mcut is not  None: self.set_mcut(mcut)

        return  potential_iso(R, Z, d0=self.d0, rc=self.rc, e=self.e, mcut=self.mcut, toll=self.toll, grid=grid)

    def _potential_parallel(self, R, Z, grid=False, toll=1e-4, mcut=None, nproc=2):
        """Calculate the potential in R and Z using a parallelized code.

        :param R: Cylindrical radius [kpc]
        :param Z: Cylindrical height [kpc]
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll: tollerance for quad integration
        :param mcut: elliptical radius where dens(m>mcut)=0
        :return:
        """

        self.set_toll(toll)

        if mcut is not None: self.set_mcut(mcut)

        pardo=ParDo(nproc=nproc)
        pardo.set_func(potential_iso)

        if len(R)!=len(Z) or grid==True:

            htab=pardo.run_grid(R,args=(Z,self.d0,self.rc,self.e,self.mcut,self.toll,grid))

        else:

            htab = pardo.run(R,Z, args=(self.d0, self.rc, self.e, self.mcut, self.toll, grid))


        return htab

