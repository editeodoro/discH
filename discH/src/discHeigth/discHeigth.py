from ..galpotential.galpotential import galpotential

class discHeigth():

    def __init__(self, fixed_component=(),external_potential=None):

        self.fixed_component=fixed_component

    def _fixed_potential(self,R,Z,grid,nproc,Rcut,zcut,mcut,toll,external_potential):

        df=galpotential(dynamic_components=self.fixed_component)
        df.potential(R,Z,grid=grid,nproc=nproc, toll=toll, Rcut=Rcut, zcut=zcut, mcut=mcut,external_potential=external_potential)
        self.fixed_potential_grid=df.potential_grid


