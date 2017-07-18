from .pot_c_ext.isothermal_halo import potential_iso
import multiprocessing as mp
from ..pardo.Pardo import ParDo

class halo:

    def __init__(self,d0,rc,e,mcut=None):

        self.d0=d0
        self.rc=rc
        self.e=e
        self.toll=1e-4
        if mcut is None:
            self.mcut=20*self.rc
        else:
            self.mcut=mcut

    def set_toll(self,toll):

        self.toll=toll

    def set_mcut(self,mcut):

        self.toll=mcut

    def potential(self,R,Z,grid=False,toll=1e-4,mcut=None, nproc=1):

        if nproc==1:
            return self._potential_serial(R=R,Z=Z,grid=grid,toll=toll,mcut=mcut)
        else:
            return self._potential_parallel(R=R, Z=Z, grid=grid, toll=toll, mcut=mcut,nproc=nproc)

    def _potential_serial(self,R,Z,grid=False,toll=1e-4,mcut=None):

        raise NotImplementedError('Potential serial not implemented for this class')

    def _potential_parallel(self, R, Z, grid=False, toll=1e-4, mcut=None,nproc=2):

        raise NotImplementedError('Potential serial not implemented for this class')

class isothermal_halo(halo):

    def __init__(self,d0,rc,e,mcut=None):

        super(isothermal_halo,self).__init__(d0=d0,rc=rc,e=e,mcut=mcut)

    def _potential_serial(self, R, Z, grid=False, toll=1e-4, mcut=None):

        self.set_toll(toll)

        if mcut is not  None: self.set_mcut(mcut)

        return  potential_iso(R, Z, d0=self.d0, rc=self.rc, e=self.e, mcut=self.mcut, toll=self.toll, grid=grid)

    def _potential_parallel(self, R, Z, grid=False, toll=1e-4, mcut=None, nproc=2):

        self.set_toll(toll)

        if mcut is not None: self.set_mcut(mcut)

        pardo=ParDo(nproc=nproc)
        pardo.set_func(potential_iso)

        htab=pardo.run(R,args=(Z,self.d0,self.rc,self.e,self.mcut,grid,self.toll))

        return htab

