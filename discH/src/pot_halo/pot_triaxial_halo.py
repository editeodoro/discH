from __future__ import division, print_function
from .pot_c_ext.triaxial_doublepower_halo import potential_triaxial_doublepower#, vcirc_powercut
from .pot_c_ext.triaxial_exponential_halo import potential_triaxial_exponential#, vcirc_powercut

import multiprocessing as mp
from ..pardo.Pardo import ParDo
import numpy as np

class triaxial_halo(object):
    """
    Super class for triaxal halo potentials
    
    These halos have 3D density profile in form of d = d(m),
    
    where m = a*sqrt(x**2/a**2 y**2/b**2 + z**2/c**2)  [BT08, eq. 2.138]
    """
    def __init__(self,d0,rc,a=1.,b=1.,c=1.,mcut=100):
        """ Common parameters for all triaxial halos

        :param d0:      Central density in Msun/kpc^3
        :param rc:      Scale radius in kpc
        :param a,b,c:   Axis ratios (see above)
        :param mcut:    Elliptical radius where dens(m>mcut)=0
        """

        self.d0=d0
        self.rc=rc
        self.a=a
        self.b=b
        self.c=c
        self.toll=1e-4
        self.mcut=mcut
        self.name='General triaxial halo'


    def potential(self,x,y,z,grid=False,toll=1e-4,mcut=None, nproc=1):
        """Calculate potential at coordinate (x,y,z). If R and Z are arrays with unequal lengths or
            if grid is True, the potential will be calculated in a 2D grid in R and Z.

        :param x,y,z: Cartesian coordinates
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll:  tollerance for quad integration
        :param mcut:  elliptical radius where dens(m>mcut)=0
        :param nproc: Number of processes
        
        :return:  An array with:
            Column 0: x
            Column 1: y
            Column 2: z
            Column 3: Potential
        """

        if not (len(x)==len(y)==len(z)) or grid:
            ndim = len(x)*len(y)*len(z)
        else:
            ndim = len(x)

        if mcut is None:
            mcut=self.mcut
        else:
            self.mcut=mcut

        if nproc==1 or ndim<100000:
            return self._potential_serial(x=x,y=y,z=z,grid=grid,toll=toll,mcut=mcut)
        else:
            return self._potential_parallel(x=x,y=y,z=z,grid=grid,toll=toll,mcut=mcut,nproc=nproc)

    def _potential_serial(self,x,y,z,grid=False,toll=1e-4,mcut=None):
        """ Specialization of potential(...) for serial calculation """
        raise NotImplementedError('Potential serial not implemented for this class')

    def _potential_parallel(self,x,y,z,grid=False, toll=1e-4, mcut=None,nproc=2):
        """ Specialization of potential(...) for serial calculation """
        raise NotImplementedError('Potential parallel not implemented for this class')

''' To implement
    def vcirc(self, R, toll=1e-4, nproc=1):
        """Calculate Vcirc at planar radius coordinate R.
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :param nproc: Number of processes
        :return:  An array with:
            0-R
            1-Vcirc
        """

        ndim=len(R)
        if nproc==1 or ndim<100000:
            return self._vcirc_serial(R=R,toll=toll)
        else:
            return self._vcirc_parallel(R=R, toll=toll, nproc=nproc)

    def _vcirc_serial(self, R, toll=1e-4):
        raise NotImplementedError('Vcirc serial not implemented for this class')


    def _vcirc_parallel(self, R, toll=1e-4, nproc=1):
        raise NotImplementedError('Potential parallel not implemented for this class')

    def dens(self, x=None, y=None, z=None, grid=False):
        """
        Evaulate the density at the point (x,y,z)
        :param x: float int or iterable
        :param y: float int or iterable
        :param z: float int or iterable, if Z is None R=m elliptical radius (m=sqrt(R*R+Z*Z/(1-e^2)) if e=0 spherical radius)
        :param grid:  if True calculate the potential in a 2D grid in R and Z, if len(R)!=len(Z) grid is True by default
        :return:  2D array with: col-0 R, col-1 dens(m) if Z is None or col-0 R, col-1 Z, col-2 dens(R,Z)
        """

        if isinstance(x, (int,float)): x = np.array([x, ])

        if x==y==z==None:
            raise ValueError('At least one between x,y,z must be defined')
        
        if y==None and z==None:
            # If y and z are not defined, return 1D array d(x,0,0)
            ret=np.zeros(shape=(len(x),2))
            ret[:,0]=x
            ret[:,1]=self._dens(x,0,0)
        
        
        
        
        else:
            if y==None:
                # If only y is not defined, return 2D array d(x,0,z)
                ret=np.zeros(shape=(len(x),2))
                ret[:,0]=x
                ret[:,1]=self._dens(x)
        

        if Z is not None:

            if isinstance(Z, int) or isinstance(Z, float):  Z = np.array([Z, ])


            if grid==True or len(R)!=len(Z):

                ret=np.zeros(shape=(len(R)*len(Z),3))

                coord=cartesian(R,Z)
                ret[:,:1]=coord
                ret[:,2]=self._dens(coord[:,0],coord[:,1])

            else:

                ret=np.zeros(shape=(len(R),3))
                ret[:,0]=R
                ret[:,1]=Z
                ret[:,2]=self._dens(R,Z)

        else:

            ret=np.zeros(shape=(len(R),2))

            ret[:,0]=R
            ret[:,1]=self._dens(R)

        return ret

    def __str__(self):

        s=''
        s+='Model: General halo\n'
        s+='d0: %.2f Msun/kpc3 \n'%self.d0
        s+='rc: %.2f\n'%self.rc
        s+='e: %.3f \n'%self.e
        s+='mcut: %.3f \n'%self.mcut

        return s
'''


class triaxial_doublepower_halo(triaxial_halo):

    def __init__(self,d0,rc,alpha=1.,beta=3.,a=1.,b=1.,c=1.,mcut=100):
        """
        Double power-law triaxial potential (e.g., Dehen potentials, BT08 eq. 2.64)
        d = d0 / ((m/rc)**(alpha) * (1+m/rc)**(beta-alpha))
        
        where m**2 = a**2(x**2/a**2 y**2/b**2 + z**2/c**2)

        Particular doublepower models (implemented below as special classes):
            - Hernquist: alpha=1, beta=4
            - Jaffe: alpha=2, beta=4
            - NFW: alpha=1, beta=3

        :param d0:          Central density in Msun/kpc^3
        :param rc:          Scale radius in kpc
        :param alpha,beta:  indexes of power laws
        :param a,b,c:       axis ratios
        :param mcut:        elliptical radius where dens(m>mcut)=0
        """

        if alpha<0 or beta<alpha:
            raise ValueError("alpha must be > 0 and beta>=alpha")    
        
        self.alpha = alpha
        self.beta = beta
        super(triaxial_doublepower_halo,self).__init__(d0=d0,rc=rc,a=a,b=b,c=c,mcut=mcut)
        self.name='Triaxial Double Power law'


    def __str__(self):
        s=''
        s+='Model: %s\n'%self.name
        s+='Density: d = d0 / ((m/rc)**(alpha) * (1+m/rc)**(beta-alpha)) \n '
        s+= '   with m**2 = x**2/a**2 + y**2/b**2 + z**2/c**2 \n'
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rc: %.2f\n'%self.rc
        s+='alpha: %.2f\n'%self.alpha
        s+='beta: %.2f\n'%self.beta
        s+='a: %.2f\n'%self.a
        s+='b: %.2f\n'%self.b
        s+='c: %.2f\n'%self.c
        s+='mcut: %.3f \n'%self.mcut
        return s
        
        
    def _dens(self, x, y=0, z=0):
        """ Return density at (x,y,z) """
        m  = self.a*np.sqrt(x**2/self.a**2+y**2/self.b**2+z**2/self.c**2)
        d  = self.d0 / ((m/self.rc)**self.alpha * (1+m/self.rc)**(self.beta-self.alpha))
        return d


    def _potential_serial(self, x, y, z, grid=False, toll=1e-4, mcut=None):
        """Calculate the potential in (x,y,z) using a serial code

        :param x,y,z: Cartesian coordinates [kpc]
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll: tollerance for quad integration
        :param mcut: elliptical radius where dens(m>mcut)=0
        """
        self.toll = toll
        return potential_triaxial_doublepower(x,y,z,self.d0,self.rc,self.alpha,self.beta,self.a,self.b,self.c,mcut,self.toll,grid)

''' To be implemented
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
        pardo.set_func(potential_triaxial_doublepower)

        if len(R)!=len(Z) or grid==True:
            htab = pardo.run_grid(R,args=(Z,self.d0,self.rc,self.rb,self.alpha,self.e,mcut,self.toll,grid))
        else:
            htab = pardo.run(R,Z, args=(self.d0,self.rc,self.rb,self.alpha,self.e,mcut,self.toll,grid))
        
        return htab


    def _vcirc_serial(self, R, toll=1e-4):
        """Calculate the Vcirc in R using a serial code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :return:
        """
        self.set_toll(toll)

        return np.array(vcirc_powercut(R,self.d0,self.rs,self.rb,self.alpha,self.e,self.toll))

    def _vcirc_parallel(self, R, toll=1e-4, nproc=1):
        """Calculate the Vcirc in R using a parallelized code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :param nproc: Number of processes
        :return:
        """

        self.set_toll(toll)
        pardo=ParDo(nproc=nproc)
        pardo.set_func(vcirc_powercut)
        htab=pardo.run_grid(R,args=(self.d0,self.rs,self.rb,self.alpha,self.e,self.toll))

        return htab

'''


class triaxial_hernquist_halo(triaxial_doublepower_halo):
    """ Hernquist halo: double power law with alpha=1, beta=4 """

    def __init__(self,d0,rc,a=1.,b=1.,c=1.,mcut=100):
        super(triaxial_hernquist_halo,self).__init__(d0=d0,rc=rc,alpha=1,beta=4,a=a,b=b,c=c,mcut=mcut)
        self.name='Hernquist model'

class triaxial_jaffe_halo(triaxial_doublepower_halo):
    """ Jaffe halo: double power law with alpha=2, beta=4 """

    def __init__(self,d0,rc,a=1.,b=1.,c=1.,mcut=100):
        super(triaxial_jaffe_halo,self).__init__(d0=d0,rc=rc,alpha=2,beta=4,a=a,b=b,c=c,mcut=mcut)
        self.name='Jaffe model'
        
class triaxial_nfw_halo(triaxial_doublepower_halo):
    """ NFW halo: double power law with alpha=1, beta=3 """

    def __init__(self,d0,rc,a=1.,b=1.,c=1.,mcut=100):
        super(triaxial_nfw_halo,self).__init__(d0=d0,rc=rc,alpha=1,beta=3,a=a,b=b,c=c,mcut=mcut)
        self.name='Navarro-Frenk_White model'


class triaxial_exponential_halo(triaxial_halo):

    def __init__(self,d0,rc,alpha=1,a=1.,b=1.,c=1.,mcut=100):
        """
        Exponential triaxial potential
        d = d0*exp(-(m/rc)**alpha)
        
        where m**2 = a**2(x**2/a**2 y**2/b**2 + z**2/c**2)

        :param d0:          Central density in Msun/kpc^3
        :param rc:          Scale radius in kpc
        :param alpha:       power of exp argument
        :param a,b,c:       axis ratios
        :param mcut:        elliptical radius where dens(m>mcut)=0
        """

        if alpha<0:
            raise ValueError("alpha must be > 0 ")    
        
        self.alpha = alpha
        super(triaxial_exponential_halo,self).__init__(d0=d0,rc=rc,a=a,b=b,c=c,mcut=mcut)
        self.name='Triaxial Exponential'


    def __str__(self):
        s=''
        s+='Model: %s\n'%self.name
        s+='Density: d = d0*exp(-(m/rc)**alpha) \n '
        s+= '   with m**2 = x**2/a**2 + y**2/b**2 + z**2/c**2 \n'
        s+='d0: %.2e Msun/kpc3 \n'%self.d0
        s+='rc: %.2f\n'%self.rc
        s+='alpha: %.2f\n'%self.alpha
        s+='a: %.2f\n'%self.a
        s+='b: %.2f\n'%self.b
        s+='c: %.2f\n'%self.c
        s+='mcut: %.3f \n'%self.mcut
        return s
        
        
    def _dens(self, x, y=0, z=0):
        """ Return density at (x,y,z) """
        m  = self.a*np.sqrt(x**2/self.a**2+y**2/self.b**2+z**2/self.c**2)
        d  = self.d0 * np.exp(-(m/self.rc)**self.alpha)
        return d


    def _potential_serial(self, x, y, z, grid=False, toll=1e-4, mcut=None):
        """Calculate the potential in (x,y,z) using a serial code

        :param x,y,z: Cartesian coordinates [kpc]
        :param grid:  if True calculate the potential in a 2D grid in R and Z
        :param toll: tollerance for quad integration
        :param mcut: elliptical radius where dens(m>mcut)=0
        """
        self.toll = toll
        return potential_triaxial_exponential(x,y,z,self.d0,self.rc,self.alpha,self.a,self.b,self.c,mcut,self.toll,grid)

''' To be implemented
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
        pardo.set_func(potential_triaxial_doublepower)

        if len(R)!=len(Z) or grid==True:
            htab = pardo.run_grid(R,args=(Z,self.d0,self.rc,self.rb,self.alpha,self.e,mcut,self.toll,grid))
        else:
            htab = pardo.run(R,Z, args=(self.d0,self.rc,self.rb,self.alpha,self.e,mcut,self.toll,grid))
        
        return htab


    def _vcirc_serial(self, R, toll=1e-4):
        """Calculate the Vcirc in R using a serial code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :return:
        """
        self.set_toll(toll)

        return np.array(vcirc_powercut(R,self.d0,self.rs,self.rb,self.alpha,self.e,self.toll))

    def _vcirc_parallel(self, R, toll=1e-4, nproc=1):
        """Calculate the Vcirc in R using a parallelized code
        :param R: Cylindrical radius [kpc]
        :param toll: tollerance for quad integration
        :param nproc: Number of processes
        :return:
        """

        self.set_toll(toll)
        pardo=ParDo(nproc=nproc)
        pardo.set_func(vcirc_powercut)
        htab=pardo.run_grid(R,args=(self.d0,self.rs,self.rb,self.alpha,self.e,self.toll))

        return htab

'''


