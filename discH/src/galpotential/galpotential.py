from __future__ import division, print_function
from ..pot_disc.pot_disc import disc
from ..pot_halo.pot_halo import  halo
import numpy as np
import time
import copy

class galpotential:


    def __init__(self,dynamic_components=()):

        self._check_components(dynamic_components)
        if isinstance(dynamic_components,list) or isinstance(dynamic_components,tuple) or isinstance(dynamic_components,np.ndarray):
            self.dynamic_components=list(dynamic_components)
        else:
            self.dynamic_components=(dynamic_components,)
        self.potential_grid=None
        self.external_potential=None
        self.potential_grid_exist=False

    def _check_components(self, components):

        if isinstance(components,list) or isinstance(components, tuple) or isinstance(components, np.ndarray):
            i=0
            for comp in components:
                if isinstance(comp, disc) or isinstance(comp, halo):
                    pass
                else:
                    raise ValueError('Dynamic components %i is not from class halo or disc'%i)
                i+=1
        elif isinstance(components, disc) or isinstance(components, halo):
            pass
        else:
            raise ValueError('Dynamic component is not from class halo or disc')

        return 0

    def add_components(self,components=()):

        self._check_components(components)

        self.dynamic_components=self.dynamic_components+list(components)

        return 0

    def remove_components(self,idx=()):

        dynamic_components=[]

        for i in range(len(self.dynamic_components)):

            for j in idx:
                if i!=j:
                    print('i',i,idx)
                    dynamic_components.append(self.dynamic_components[i])
                else:
                    pass

        self.dynamic_components=dynamic_components

        return 0

    def _make_finalgrid(self,R,Z,ncolumn=3,grid=False):

        lenR=len(R)
        lenZ=len(Z)

        if lenR!=lenZ or grid==True:
            nrow=lenR*lenZ
        else:
            nrow=lenR

        arr=np.zeros(shape=(nrow,ncolumn))

        return arr

    def potential(self,R,Z,grid=False,nproc=2, toll=1e-4, Rcut=None, zcut=None, mcut=None,external_potential=None):

        grid_final=self._make_finalgrid(R,Z,ncolumn=3,grid=grid)
        grid_complete=self._make_finalgrid(R,Z,ncolumn=len(self.dynamic_components)+4,grid=grid)
        self.external_potential=external_potential

        #External potential
        print('External potential: ',flush=True,end='')
        if external_potential is not None:
            if len(external_potential)!=len(grid_final):
                raise ValueError('External potential dimension (%i) are than the user defined grid dimension (%i)'%(len(external_potential),len(grid_final)))
            else:
                grid_complete[:,-2]=external_potential[:,-1]
                grid_final[:,-1]=external_potential[:,-1]
                print('Yes',flush=True)
        else:
            print('No',flush=True)

        #Calc potential
        i=0
        for comp in self.dynamic_components:
            print('Calculating Potential of the %ith component (%s)...'%(i+1,comp.name),end='',flush=True)
            if isinstance(comp, halo):
                tini=time.time()
                grid_tmp = comp.potential(R, Z, grid=grid, toll=toll, mcut=mcut, nproc=nproc)
                tfin=time.time()
            elif isinstance(comp,disc):
                tini=time.time()
                grid_tmp = comp.potential(R,Z,grid=grid,toll=toll,Rcut=Rcut, zcut=zcut, nproc=nproc)
                tfin=time.time()
            tottime=tfin-tini
            print('Done (%.2f s)'%tottime)
            if i==0:
                grid_final[:,0]=grid_tmp[:,0]
                grid_final[:,1]=grid_tmp[:,1]
                grid_final[:,2]+=grid_tmp[:,2]
                grid_complete[:,0]=grid_tmp[:,0]
                grid_complete[:,1]=grid_tmp[:,1]
                grid_complete[:,2]=grid_tmp[:,2]
            else:
                grid_final[:,2]+=grid_tmp[:,2]
                grid_complete[:,2+i]=grid_tmp[:,2]
            i+=1
        grid_complete[:,-1]=np.sum(grid_complete[:,2:-2],axis=1)

        self.potential_grid=grid_final
        self.potential_grid_complete=grid_complete
        self.potential_grid_exist=True
        self.dynamic_components_last=copy.copy(self.dynamic_components)

        return grid_final

    def save(self,filename,complete=True):


        if complete: save_arr=self.potential_grid_complete
        else: save_arr=self.potential_grid

        if self.potential_grid_exist:

            if complete:
                header=''
                header+='0-R 1-Z'

                i=2
                for comp in self.dynamic_components_last:
                    header+=' %i-%s'%(i,comp.name)
                    i+=1
                header+=' %i-External %i-Total'%(i,i+1)
                save_arr = self.potential_grid_complete

            else:

                header='0-R 1-Z 2-Total'
                save_arr = self.potential_grid

        else:

            raise AttributeError('Potential grid does not exist, make it with potential method')

        footer='R and Z in Kpc, Potentials in Kpc^2/Myr^2\n'
        i=0
        for comp in self.dynamic_components_last:
            footer += '*****************\n'
            footer += 'Component %i \n'%i
            footer += comp.__str__()
            i += 1
        footer += '*****************'

        np.savetxt(filename,save_arr,fmt='%.5e',header=header,footer=footer)




    def dynamic_components_info(self):

        i=0
        for comp in self.dynamic_components:

            print('Components:',i)
            print(comp)
            i+=1