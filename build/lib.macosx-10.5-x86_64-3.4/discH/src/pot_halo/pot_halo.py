from .pot_c_ext.isothermal_halo import potential_iso
import multiprocessing as mp
from ..pardo.Pardo import ParDo

def _potential_isothermal_parallel(R,Z,parproc,d0,rc,e,mcut,grid=False,toll=1e-4):

    parproc.output.put(potential_iso(R, Z, d0=d0, rc=rc, e=e, mcut=mcut, toll=toll, grid=grid))

class isothermal_halo:

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

    def potential(self,R,Z,grid=False,toll=1e-4):

        self.set_toll(toll)

        return  potential_iso(R, Z, d0=self.d0, rc=self.rc, e=self.e, mcut=self.mcut, toll=self.toll, grid=grid)

    def potential_parallel(self,R,Z,grid=False, toll=1e-4, nproc=2):

        pardo=ParDo(nproc=nproc)

        htab=pardo.run(R,_potential_isothermal_parallel,args=(Z,pardo,self.d0,self.rc,self.e,self.mcut,grid,toll))

        return htab