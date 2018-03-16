from __future__ import print_function
import sys, time 
from .galpotential import galpotential
from ..pot_disc.pot_disc import *
from ..pot_halo.pot_halo import *
from ..pot_halo.pot_triaxial_halo import *
from ..utils import *

try:
    from scipy.interpolate import interp1d
    import numpy as np
except:
    raise ImportError('Scipy and Numpy modules are needed. Install them!')


class MWpotential(galpotential):
    """ 
    This class implements the most used potentials for the Milky Way.
    Current implemented potentials are:
    1) Binney&Tremaine08, Model 1 (Tab 2.3)
    2) Binney&Tremaine08, Model 2 (Tab 2.3)
    3) Sormani et al. 2017
    4) McMillan 2017
    """
    
    def __init__(self,model='BT08_Model1'):
        
        # Knowns models: Binney&Tremaine08 models 1 and 2, Sormani+17, McMillan17
        self.models = ['BT08_Model1', 'BT08_Model2', 'S+17', 'McM17']
        if model in '\t'.join(self.models):
            self.mod = model
        else:
            raise ValueError("model must be either 'BT08_Model1', 'BT08_Model2', S+17 or McM17'")
        
        # Grid where the potential is defined. Can be a 2D or a 3D array
        self.potgrid = None
        # The total potential. Can be a 2D or a 3D array
        self.totalpot = None
        # This is an array containing the potentials of single components
        self.pots = None
        # The total circular velocity. 
        self.vcirctot = None
        # An array of circular velocities of different components
        self.vcircs = None
        
        dc = self._getComponents()
        
        # Number of dynamical components
        self.ncomps = len(dc)
        
        super(MWpotential,self).__init__(dynamic_components=dc)
        
        
    def _getComponents (self):
        
        if 'BT08' in self.mod:
            """ Binney&Tremaine08 Model 1 or 2
            
            The potential is made by the following components:
            1) BULGE (oblate ellipsoid):
                rho_b = rho0_b / (m/rc)**alpha * exp(-(m/rb)**beta) 
                where m = sqrt(x**2+y**2+z**2/q**2)
            2) DISK (stellar):
                rho_d = sig1/(2z1)*exp(-z/z1-R/R1) + sig2/(2z2)*exp(-z/z2-R/R2)
            2) DISK (gas) with central hole:
                rho_d = sig1/(2z1)*exp(-z/z1-R/R1-Rm/R)
            3) HALO (double power):
                rho_h = rho0_h / ((m/rh)**(alpha)*(1+m/rh)**(beta-alpha))
            """
            
            m = 1 if "1" in self.mod else 2
            
            # BULGE ----------------------------------------
            q_b    = 0.6
            e_b    = np.sqrt(1-q_b**2)
            rscl_b = 1.       # kpc
            rcut_b = 1.9      # kpc
            alph_b = 1.8    
            rho0_b = 0.427E09 if m==1 else 0.3E09# Msun/kpc3
            
            bulge = powercut_halo(d0=rho0_b,rc=rscl_b,rb=rcut_b,alpha=alph_b,e=e_b)
            #-----------------------------------------------

            # HALO ----------------------------------------
            q_h    = 0.8
            e_h    = np.sqrt(1-q_h**2) 
            if m==1:
                rho0_h = 0.711E09 # Msun/kpc3
                rscl_h = 3.83     # kpc
                alph_h = -2    
                beta_h = 2.96    
            else:
                rho0_h = 0.266E09 # Msun/kpc3
                rscl_h = 1.90     # kpc
                alph_h = 1.63    
                beta_h = 2.17

            halo = alfabeta_halo(d0=rho0_h,rs=rscl_h,alpha=alph_h,beta=beta_h,e=e_h)
            #-----------------------------------------------
            
            # DISK (thin+thick stellar + gas) --------------
            zd_thin  = 0.3  # kpc
            zd_thick = 1.   # kpc
            rscl_d = 2. if m==1 else 3.2 # kpc
            rs_gas = 2*rscl_d
            rm_gas = 4
            zd_gas = 0.08
            # BT08 do not give the sigma0 for the various components, but gives sigma0_star + sigma0_gas
            # and says that at Rsun = 8 kpc we have sigma_gas = 0.25*sigma_tot, 
            # sigma_star_thin = 0.75*0.95*sigma_tot and sigma_star_thin = 0.75*0.05*sigma_tot and 
            # Solving this nice riddle...
            sigma0_tot = 1.905E09 if m==1 else 0.536E09 # Msun/kpc2
            R0 = 8.
            alpha = np.exp(-R0/rs_gas-rm_gas/R0)
            beta = np.exp(-R0/rscl_d)
            sig0_star = sigma0_tot / (1+beta/(3*alpha))
            sig0_gas  = sigma0_tot - sig0_star
            sig0_star_thick = sig0_star/21.
            sig0_star_thin  = sig0_star - sig0_star_thick
            
            disk_thin  = Exponential_disc.thick(sigma0=sig0_star_thin, Rd=rscl_d, zd=zd_thin, zlaw='exp')
            disk_thick = Exponential_disc.thick(sigma0=sig0_star_thick, Rd=rscl_d, zd=zd_thick, zlaw='exp')
            disk_gas   = Exponential_disc.thick(sigma0=sig0_gas, Rd=rs_gas, Rm=rm_gas, zd=zd_gas, zlaw='exp')
            #-----------------------------------------------
            
            dc = [bulge,halo,disk_thin,disk_thick,disk_gas]
        
        elif 'S+17' in self.mod:
            """ Sormani et al (2017) potential
            
            The potential is made by the following components:
            1) BULGE (oblate ellipsoid):
                rho_b = rho0_b / (m/rc)**alpha * exp(-(m/rb)**beta) 
                where m = sqrt(x**2+y**2+z**2/q**2)
            2) DISK (stellar+gas):
                rho_d = sig1/(2z1)*exp(-z/z1-R/R1) + sig2/(2z2)*exp(-z/z2-R/R2)
            3) HALO (spherical NFW):
                rho_h = rho0_h / ((r/rh)*(1+r/rh)**2)
            4) BAR (prolate exponential ellipsoid)
                rho_bar = rho0_bar * exp(-m/rc)
                where m = sqrt(x**2+(y**2+z**2)/q**2)
            """
            # BULGE ----------------------------------------
            q_b    = 0.5
            e_b    = np.sqrt(1-q_b**2)
            rho0_b = 0.8E09     # Msun/kpc3
            rscl_b = rcut_b = 1 # kpc
            alph_b = 1.7    
            
            bulge = powercut_halo(d0=rho0_b,rc=rscl_b,rb=rcut_b,alpha=alph_b,e=e_b)
            #-----------------------------------------------
            
            # HALO ----------------------------------------
            rho0_h = 8.11E06  # Msun/kpc3
            rscl_h = 19.6     # kpc  
            
            halo = alfabeta_halo(d0=rho0_h,rs=rscl_h,alpha=1,beta=3,e=0)
            #-----------------------------------------------
            
            # DISK (thin+thick) ----------------------------
            zd_thin   = 0.3    # kpc
            rscl_thin = 2.5    # kpc 
            sig0_thin = 850E06 # Msun/kpc2
            disk_thin = Exponential_disc.thick(sigma0=sig0_thin, Rd=rscl_thin, zd=zd_thin, zlaw='exp')
            
            zd_thick   = 0.9    # kpc
            rscl_thick = 3.02   # kpc 
            sig0_thick = 174E06 # Msun/kpc2
            disk_thick = Exponential_disc.thick(sigma0=sig0_thick, Rd=rscl_thick, zd=zd_thick, zlaw='exp')
            #-----------------------------------------------
            
            # BAR ------------------------------------------
            rho0_bar = 5E09  # Msun/kpc3
            rscl_bar = 0.75  # kpc
            q_bar    = 0.5
            bar = triaxial_exponential_halo(d0=rho0_bar,rc=rscl_bar,alpha=1,a=1,b=q_bar,c=q_bar) 
            #-----------------------------------------------
            
            dc = [bulge,halo,disk_thin,disk_thick,bar]
            
        elif 'McM17' in self.mod:            
            """ McMillan 2017 potential
            
            The potential is made by the following components:
            1) BULGE (oblate ellipsoid):
                rho_b = rho0_b / (1+m/rc)**alpha * exp(-(m/rb)**2) 
                where m = sqrt(x**2+y**2+z**2/q**2)
            2) DISK (stellar thick and thin + HI + H2):
                rho_d_star = sig1/(2z1)*exp(-z/z1-R/R1) 
            2) DISK (gas) with central hole:
                rho_d = sig1/(4z1)*exp(-R/R1-Rm/R)*sech(z/2z1)
            3) HALO (nfw):
                rho_h = rho0_h / ((m/rh)*(1+m/rh)**2)
            """
            
            # BULGE ----------------------------------------
            q_b    = 0.5
            e_b    = np.sqrt(1-q_b**2)
            rho0_b = 98.4E09    # Msun/kpc3
            rscl_b = 0.075 # kpc
            rcut_b = 2.1   # kpc
            alph_b = 1.8    
            
            # THIS IS NOT CORRECT
            bulge = powercut_halo(d0=rho0_b,rc=rscl_b,rb=rcut_b,alpha=alph_b,e=e_b)
            #-----------------------------------------------
            
            # DISK (thin+thick stellar + HI+H2 gas) --------------
            sig0_thin  = 896E06  # Mpc/kpc2
            zd_thin    = 0.3     # kpc
            rs_thin    = 2.5     # kpc 
            sig0_thick = 183E06  # Mpc/kpc2
            zd_thick   = 0.9     # kpc
            rs_thick   = 3.02    # kpc      
            sig0_HI    = 53.1E06 # Mpc/kpc2
            rs_HI      = 7.      # kpc 
            rm_HI      = 4.      # kpc 
            zd_HI      = 0.17    # kpc
            sig0_H2    = 2180E06 # Mpc/kpc2
            rs_H2      = 1.5     # kpc 
            rm_H2      = 12.     # kpc 
            zd_H2      = 0.09    # kpc

            disk_thin  = Exponential_disc.thick(sigma0=sig0_thin, Rd=rs_thin, zd=zd_thin, zlaw='exp')
            disk_thick = Exponential_disc.thick(sigma0=sig0_thick, Rd=rs_thick, zd=zd_thick, zlaw='exp')
            disk_HI    = Exponential_disc.thick(sigma0=sig0_HI, Rd=rs_HI, Rm=rm_HI, zd=zd_HI, zlaw='sech2')
            disk_H2    = Exponential_disc.thick(sigma0=sig0_H2, Rd=rs_H2, Rm=rm_H2, zd=zd_H2, zlaw='sech2')
            #-----------------------------------------------
            
            # HALO ----------------------------------------
            rho0_h = 8.54E06  # Msun/kpc3
            rscl_h = 19.6     # kpc  
            
            halo = alfabeta_halo(d0=rho0_h,rs=rscl_h,alpha=1,beta=3,e=0)
            #-----------------------------------------------
            
            dc = [bulge,halo,disk_thin,disk_thick,disk_HI,disk_H2]
            
        return dc
        
        
    def calculate_potential(self,coordgrid,grid=False,nproc=2, toll=1e-4, Rcut=None, zcut=None, mcut=None,external_potential=None):
        
        """ Calculate the potential for the MW and write it on self.totalpot and self.pots
        
        :param coordgrid:         tuple, can be (R,z) or (x,y,z)
        :param grid:              if True, calculate phi on a full grid (R,z)
        :param toll:              float, tolerance for integration
        :param Rcut,zcut,mcut     Cut for density, e.g. rho(R>Rcut)=0
        :param external_potential Additional potential to add to the model
        
        :return (self.totalpot, self.pots)
        """
        
        if "BT08" in self.mod:        
            """ If the potential is axisymmetric, we can use the implemention of the 
                parent class """
            
            coordgrid = np.array(coordgrid)
            if coordgrid.shape[0]!=2:
                raise ValueError("For BT08 models, coordgrid must be a 2D array (R,z)")
                
            R, Z = coordgrid
                        
            super(MWpotential,self).potential(R=R,Z=Z,grid=grid,nproc=nproc,toll=toll,Rcut=Rcut,\
                                              zcut=zcut,mcut=mcut,external_potential=None)

            pc = self.potential_grid_complete
            pot_b, pot_h, pot_d1, pot_d2, pot_d3 = pc[:,2], pc[:,3], pc[:,4], pc[:,5], pc[:,6]

            # Put potentials on (R,z) grid. Axis order is [z,R]!!!
            pt_b  = pot_b.reshape(len(R),len(Z)).T
            pt_h  = pot_h.reshape(len(R),len(Z)).T
            pt_d1 = pot_d1.reshape(len(R),len(Z)).T
            pt_d2 = pot_d2.reshape(len(R),len(Z)).T
            pt_d3 = pot_d3.reshape(len(R),len(Z)).T
            
            self.potgrid = np.array([R,Z])
            self.totalpot = pc[:,-1].reshape(len(R),len(Z)).T 
            self.pots = np.array([pt_b,pt_h,pt_d1,pt_d2,pt_d3])
                        
        elif 'S+17' in self.mod:
            
            coordgrid = np.array(coordgrid)
            if coordgrid.shape[0]!=3:
                raise ValueError("For Sormani+17 model, coordgrid must be a 3D array (x,y,z)")
            
            X, Y, Z = coordgrid
            
            """ First we integrate the axisymmetric part of the potential (disk+bulge+halo) """
            
            # Considering only x and y to get the R-grid.
            xpos, ypos = X[X>=0], Y[Y>=0]
            # If x and y have different sizes, interpolate one
            if len(xpos)>len(ypos):
                f1 = interp1d(np.linspace(0,1,len(ypos)),ypos,kind='linear')
                xp = xpos
                yp = f1(np.linspace(0,1,len(xpos)))
            elif len(xpos)<len(ypos):
                f1 = interp1d(np.linspace(0,1,len(xpos)),xpos,kind='linear')
                xp = f1(np.linspace(0,1,len(ypos)))
                yp = ypos
            else: xp, yp = xpos, ypos
    
            R = np.sqrt(xp**2+yp**2)
        
            # Dummy galpotential object with axisymmetric part
            axisym = galpotential(dynamic_components=self.dynamic_components[:-1])
            axisym.potential(R=R,Z=Z,grid=grid,nproc=nproc,toll=toll,Rcut=Rcut,\
                             zcut=zcut,mcut=mcut,external_potential=None)
            
            pc = axisym.potential_grid_complete
            pot_b, pot_h, pot_d1, pot_d2 = pc[:,2], pc[:,3], pc[:,4], pc[:,5]
            pt_b  = pot_b.reshape(len(R),len(Z)).T
            pt_h  = pot_h.reshape(len(R),len(Z)).T
            pt_d1 = pot_d1.reshape(len(R),len(Z)).T
            pt_d2 = pot_d2.reshape(len(R),len(Z)).T
            
            # Interpolating the axysimmetric (R,z) values on a (x,y,z) grid
            print('Regridding axisymmetric components...',end='')
            tini = time.time()
            
            pt_b_xyz  = interp_2D_to_3D(grid2D=(R,Z),pot2D=pt_b,grid3D=(X,Y,Z))
            pt_h_xyz  = interp_2D_to_3D(grid2D=(R,Z),pot2D=pt_h,grid3D=(X,Y,Z))
            pt_d1_xyz = interp_2D_to_3D(grid2D=(R,Z),pot2D=pt_d1,grid3D=(X,Y,Z))
            pt_d2_xyz = interp_2D_to_3D(grid2D=(R,Z),pot2D=pt_d2,grid3D=(X,Y,Z))
            
            print('Done (%.2f s)'%(time.time()-tini))
            
            
            """ Now integrating the bar """
            
            bar = self.dynamic_components[-1]
            print('Calculating Potential of the 5th component (%s)...'%(bar.name),end='')            
            tini = time.time()
            pot_bar = bar.potential(X,Y,Z,grid=grid,mcut=mcut,toll=toll,nproc=nproc)
            pt_bar  = pot_bar[:,3].reshape(len(Z),len(Y),len(X))
            print('Done (%.2f s)'%(time.time()-tini))
            
            
            # Now putting all togheter            
            self.potgrid = np.array([X,Y,Z])
            self.totalpot = pt_b_xyz + pt_h_xyz + pt_d1_xyz + pt_d2_xyz + pt_bar
            self.pots = np.array([pt_b_xyz,pt_h_xyz,pt_d1_xyz,pt_d2_xyz,pt_bar])
        
        elif "McM17" in self.mod:        
            """ If the potential is axisymmetric, we can use the implemention of the 
                parent class """
            
            coordgrid = np.array(coordgrid)
            if coordgrid.shape[0]!=2:
                raise ValueError("For BT08 models, coordgrid must be a 2D array (R,z)")
                
            R, Z = coordgrid
                        
            super(MWpotential,self).potential(R=R,Z=Z,grid=grid,nproc=nproc,toll=toll,Rcut=Rcut,\
                                              zcut=zcut,mcut=mcut,external_potential=None)

            pc = self.potential_grid_complete
            pot_b, pot_h, pot_d1, pot_d2, pot_d3, pot_d4 = pc[:,2], pc[:,3], pc[:,4], pc[:,5], pc[:,6], pc[:,7]

            # Put potentials on (R,z) grid. Axis order is [z,R]!!!
            pt_b  = pot_b.reshape(len(R),len(Z)).T
            pt_h  = pot_h.reshape(len(R),len(Z)).T
            pt_d1 = pot_d1.reshape(len(R),len(Z)).T
            pt_d2 = pot_d2.reshape(len(R),len(Z)).T
            pt_d3 = pot_d3.reshape(len(R),len(Z)).T
            pt_d4 = pot_d4.reshape(len(R),len(Z)).T
            
            self.potgrid = np.array([R,Z])
            self.totalpot = pc[:,-1].reshape(len(R),len(Z)).T 
            self.pots = np.array([pt_b,pt_h,pt_d1,pt_d2,pt_d3,pt_d4])
        
        return self.totalpot, self.pots


    def calculate_vcirc(self,R,nproc=2,toll=1e-4):

        dc = self.dynamic_components
        if self.potgrid is None: self.potgrid = np.array([R,np.zeros_like(R)])   
        
        if "BT08" in self.mod:
            # Bulge
            vc_b = dc[0].vcirc(R,nproc=nproc,toll=toll)[:,1]
            # Halo
            vc_h = dc[1].vcirc(R,nproc=nproc,toll=toll)[:,1]
            # Disk
            vc_thin  = dc[2].vcirc(R, nproc=nproc)[:,1]
            vc_thick = dc[3].vcirc(R, nproc=nproc)[:,1]
            vc_gas   = dc[4].vcirc(R, nproc=nproc)[:,1]
            vc_d = np.sqrt(vc_thin**2+vc_thick**2+vc_gas**2)
            # Total
            self.vcirctot = np.sqrt((vc_b**2+vc_h**2+vc_d**2))
            self.vcircs = [vc_b,vc_h,vc_thin,vc_thick,vc_gas]
             
        elif 'S+17' in self.mod:
            # Bulge
            vc_b = dc[0].vcirc(R,nproc=nproc,toll=toll)[:,1]
            # Halo
            vc_h = dc[1].vcirc(R,nproc=nproc,toll=toll)[:,1]
            # Disk
            vc_thin  = dc[2].vcirc(R, nproc=nproc)[:,1]
            vc_thick = dc[3].vcirc(R, nproc=nproc)[:,1]
            vc_d = np.sqrt(vc_thin**2+vc_thick**2)
            
            """ NEED TO FIND A WAY TO CALCULATE VCIRC FOR THE TRIAXIAL PART """
            vc_bar = np.zeros_like(R)
         
            # Total
            self.vcirctot = np.sqrt(vc_b**2+vc_h**2+vc_d**2+vc_bar**2)
            self.vcircs = [vc_b,vc_h,vc_thin,vc_thick,vc_bar]
         
        elif "McM17" in self.mod:
            # Bulge
            vc_b = dc[0].vcirc(R,nproc=nproc,toll=toll)[:,1]
            # Halo
            vc_h = dc[1].vcirc(R,nproc=nproc,toll=toll)[:,1]
            # Disk
            vc_thin  = dc[2].vcirc(R, nproc=nproc)[:,1]
            vc_thick = dc[3].vcirc(R, nproc=nproc)[:,1]
            vc_HI   = dc[4].vcirc(R, nproc=nproc)[:,1]
            vc_H2   = dc[5].vcirc(R, nproc=nproc)[:,1]
            vc_d = np.sqrt(vc_thin**2+vc_thick**2+vc_HI**2+vc_H2**2)
            # Total
            self.vcirctot = np.sqrt((vc_b**2+vc_h**2+vc_d**2))
            self.vcircs = [vc_b,vc_h,vc_thin,vc_thick,vc_HI,vc_H2]
         
        return self.vcirctot, self.vcircs
    
    
    def writeFITS(self,fname=None):
        
        if self.totalpot is None:
            raise RuntimeError("You must run calculate_potential() before calling writeFITS().")
        
        if "BT08" in self.mod:
            pt_d = self.pots[2]+self.pots[3]+self.pots[4]
            pots = [self.totalpot,self.pots[0],self.pots[1],pt_d]
            text = ["TOTAL","BULGE","HALO","DISK"]
            outname = 'BTpot_mod%s.fits'%('1' if '1' in self.mod else '2') 
        elif 'S+17' in self.mod:
            pt_d = self.pots[2]+self.pots[3]
            pots = [self.totalpot,self.pots[0],self.pots[1],pt_d,self.pots[-1]]
            text = ["TOTAL","BULGE","HALO","DISK","BAR"]
            outname = "Sormani+17_pot.fits"
        elif "McM17" in self.mod:
            pt_d = self.pots[2]+self.pots[3]+self.pots[4]+self.pots[5]
            pots = [self.totalpot,self.pots[0],self.pots[1],pt_d]
            text = ["TOTAL","BULGE","HALO","DISK"]
            outname = 'McM17pot_mod.fits'

        if fname: outname = fname
        
        return writeFITS(coordgrid=self.potgrid,potentials=pots,npots=len(pots),names=text,fname=outname)


    def plot_potentials(self, fname=''):
        
        if self.totalpot is None:
            raise RuntimeError("You must run calculate_potential() before calling plot_potentials().")
        
        if "BT08" in self.mod:
            pt_d = self.pots[2]+self.pots[3]+self.pots[4]
            pots = [self.totalpot,self.pots[0],self.pots[1],pt_d]
            outname = 'BTpot_mod%s.pdf'%('1' if '1' in self.mod else '2')
            text = ["TOTAL","BULGE","HALO","DISK"]
            R, Z = self.potgrid
        elif 'S+17' in self.mod:
            # Plotting x-z in the plane y=0
            Y = self.potgrid[1]
            idx = (np.abs(Y)).argmin()
            pt_d = self.pots[2]+self.pots[3]
            pots = [self.totalpot[:,idx,:],self.pots[0][:,idx,:],\
                    self.pots[1][:,idx,:],pt_d[:,idx,:],self.pots[-1][:,idx,:]]
            text = ["TOTAL","BULGE","HALO","DISK","BAR"]
            outname = "Sormani+17_pot.pdf"
            R, Z = self.potgrid[0], self.potgrid[2]
        elif "McM17" in self.mod:
            pt_d = self.pots[2]+self.pots[3]+self.pots[4]+self.pots[5]
            pots = [self.totalpot,self.pots[0],self.pots[1],pt_d]
            outname = 'McM17pot_mod.pdf'
            text = ["TOTAL","BULGE","HALO","DISK"]
            R, Z = self.potgrid
        

        if fname!='': outname = fname
        v = np.linspace(-50,-0.05,100)/100.            
        
        return plot_potentials(R=R,Z=Z,potentials=pots,npots=len(pots),names=text,contours=v,fname=outname)
        
        
    def plot_vcirc(self,fname=''):
        
        if self.vcirctot is None:
            raise RuntimeError("You must run calculate_vcirc() before calling plot_vcirc().")
        
        if "BT08" in self.mod:        
            v_d = np.sqrt(self.vcircs[2]**2+self.vcircs[3]**2+self.vcircs[4]**2)
            vcs = [self.vcirctot,self.vcircs[0],self.vcircs[1],v_d]
            outname = 'BTvcirc_mod%s.pdf'%('1' if '1' in self.mod else '2')
            ltype = ['-','--',':','-.']
            text = ["TOTAL","BULGE","HALO","DISK"]
            R = self.potgrid[0]
        elif 'S+17' in self.mod:
            v_d = np.sqrt(self.vcircs[2]**2+self.vcircs[3]**2)
            vcs = [self.vcirctot,self.vcircs[0],self.vcircs[1],v_d,self.vcircs[-1]]
            outname = 'Sormani_vcirc.pdf'
            ltype = ['-','--',':','-.', '.']
            text = ["TOTAL","BULGE","HALO","DISK","BAR"]
            R = self.potgrid[0]
        elif "McM17" in self.mod:        
            v_d = np.sqrt(self.vcircs[2]**2+self.vcircs[3]**2+self.vcircs[4]**2+self.vcircs[5]**2)
            vcs = [self.vcirctot,self.vcircs[0],self.vcircs[1],v_d]
            outname = 'McM17vcirc_mod.pdf'
            ltype = ['-','--',':','-.']
            text = ["TOTAL","BULGE","HALO","DISK"]
            R = self.potgrid[0]
        
        if fname!='': outname = fname
        
        return plot_vcirc(R=R,vcircs=vcs,nvcirc=len(vcs),names=text,ltype=ltype,fname=outname)
                