from __future__ import division, print_function
from .pot_c_ext.isothermal_halo import potential_iso
from .pot_c_ext.nfw_halo import potential_nfw
from .pot_c_ext.alfabeta_halo import potential_alfabeta
from .pot_c_ext.plummer_halo import potential_plummer
import multiprocessing as mp
from ..pardo.Pardo import ParDo
import numpy as np

class halo:
    """
    Super class for halo potentials
    """
    def __init__(self,d0,rc,e=0,mcut=100):
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
        self.mcut=mcut
        self.name='General halo'

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

        self.mcut=mcut

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

        if len(R) != len(Z) or grid == True:
            ndim = len(R) * len(Z)
        else:
            ndim = len(R)

        if mcut is None:
            mcut=self.mcut
        else:
            self.mcut=mcut

        if nproc==1 or ndim<100000:
            return self._potential_serial(R=R,Z=Z,grid=grid,toll=toll,mcut=mcut)
        else:
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


    def __str__(self):

        s=''
        s+='Model: General halo\n'
        s+='d0: %.2f Msun/kpc3 \n'%self.d0
        s+='rc: %.2f\n'%self.rc
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s

class isothermal_halo(halo):

    def __init__(self,d0,rc,e=0,mcut=100):
        """Isothermal halo d=d0/(1+r^2/rc^2)

        :param d0:  Central density in Msun/kpc^3
        :param rc:  Scale radius in kpc
        :param e:  eccentricity (sqrt(1-b^2/a^2))
        :param mcut: elliptical radius where dens(m>mcut)=0
        """
        super(isothermal_halo,self).__init__(d0=d0,rc=rc,e=e,mcut=mcut)
        self.name='Isothermal halo'

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


        return  potential_iso(R, Z, d0=self.d0, rc=self.rc, e=self.e, mcut=mcut, toll=self.toll, grid=grid)

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


        pardo=ParDo(nproc=nproc)
        pardo.set_func(potential_iso)

        if len(R)!=len(Z) or grid==True:

            htab=pardo.run_grid(R,args=(Z,self.d0,self.rc,self.e, mcut,self.toll,grid))

        else:

            htab = pardo.run(R,Z, args=(self.d0, self.rc, self.e, mcut, self.toll, grid))


        return htab

    def __str__(self):

        s=''
        s+='Model: Isothermal halo\n'
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rc: %.2f\n'%self.rc
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s

class NFW_halo(halo):

    def __init__(self,d0,rs,e=0,mcut=100):
        """NFW halo d=d0/((r/rs)(1+r/rs)^2)

        :param d0:  Central density in Msun/kpc^3
        :param rs:  Scale radius in kpc
        :param e:  eccentricity (sqrt(1-b^2/a^2))
        :param mcut: elliptical radius where dens(m>mcut)=0
        """

        self.rs=rs
        super(NFW_halo,self).__init__(d0=d0,rc=rs,e=e,mcut=mcut)
        self.name='NFW halo'

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


        return  potential_nfw(R, Z, d0=self.d0, rs=self.rc, e=self.e, mcut=mcut, toll=self.toll, grid=grid)

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


        pardo=ParDo(nproc=nproc)
        pardo.set_func(potential_nfw)

        if len(R)!=len(Z) or grid==True:

            htab=pardo.run_grid(R,args=(Z,self.d0,self.rc,self.e, mcut,self.toll,grid))

        else:

            htab = pardo.run(R,Z, args=(self.d0, self.rc, self.e, mcut, self.toll, grid))


        return htab

    def __str__(self):

        s=''
        s+='Model: NFW halo\n'
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rs: %.2f\n'%self.rs
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s


class alfabeta_halo(halo):

    def __init__(self,d0,rs,alfa,beta,e=0,mcut=100):

        if alfa>=2:
            raise ValueError('alpha must be <2')

        self.rs=rs
        self.alfa=alfa
        self.beta=beta
        super(alfabeta_halo,self).__init__(d0=d0,rc=rs,e=e,mcut=mcut)
        self.name='AlfaBeta halo'

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

        return potential_alfabeta(R, Z, d0=self.d0, alfa=self.alfa, beta=self.beta, rc=self.rc, e=self.e, mcut=mcut, toll=self.toll, grid=grid)

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


        pardo=ParDo(nproc=nproc)
        pardo.set_func(potential_alfabeta)

        if len(R)!=len(Z) or grid==True:

            htab=pardo.run_grid(R,args=(Z,self.d0,self.alfa,self.beta,self.rc,self.e, mcut,self.toll,grid))

        else:

            htab = pardo.run(R,Z, args=(self.d0, self.alfa, self.beta, self.rc, self.e, mcut, self.toll, grid))


        return htab

    def __str__(self):

        s=''
        s+='Model: %s\n'%self.name
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rs: %.2f\n'%self.rs
        s+='alfa: %.1f\n'%self.alfa
        s+='alfa: %.1f\n'%self.beta
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s

class hernquist_halo(alfabeta_halo):

    def __init__(self,d0,rs,e=0,mcut=100):

        alfa=1
        beta=4
        super(hernquist_halo,self).__init__(d0=d0,rs=rs,alfa=alfa,beta=beta,e=e,mcut=mcut)
        self.name='Hernquist halo'

    def __str__(self):

        s=''
        s+='Model: %s\n'%self.name
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rs: %.2f\n'%self.rs
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s

class deVacouler_like_halo(alfabeta_halo):

    def __init__(self,d0,rs,e=0,mcut=100):
        alfa=1.5
        beta=4
        super(deVacouler_like_halo, self).__init__(d0=d0, rs=rs, alfa=alfa, beta=beta, e=e, mcut=mcut)
        self.name = 'deVacouler like halo'

    def __str__(self):

        s=''
        s+='Model: %s\n'%self.name
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rs: %.2f\n'%self.rs
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s

class plummer_halo(halo):

    def __init__(self,rc,d0=None,mass=None,e=0,mcut=100):

        if (d0 is None) and (mass is None):
            raise ValueError('d0 or mass must be set')
        elif mass is None:
            mass=d0*(3./4.)*np.pi*rc*rc*rc
        else:
            d0=(4./3.)*mass/(np.pi*rc*rc*rc)

        super(plummer_halo,self).__init__(d0=d0,rc=rc,e=e,mcut=mcut)
        self.name='Plummer halo'
        self.mass=mass

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

        return potential_plummer(R, Z, d0=self.d0, rc=self.rc, e=self.e, mcut=mcut, toll=self.toll, grid=grid)

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


        pardo=ParDo(nproc=nproc)
        pardo.set_func(potential_plummer())

        if len(R)!=len(Z) or grid==True:

            htab=pardo.run_grid(R,args=(Z,self.d0, self.rc,self.e, mcut,self.toll,grid))

        else:

            htab = pardo.run(R,Z, args=(self.d0, self.rc, self.e, mcut, self.toll, grid))


        return htab


    def __str__(self):

        s=''
        s+='Model: %s\n'%self.name
        s+='Mass: %.2e Msun \n'%self.mass
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rc: %.2f\n'%self.rc
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s