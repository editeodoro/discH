""" 
Utility function for dealing with potentials.
Functions:

1) calculate_acceleration
2) writeFITS
3) writeHDF5
4) plot_potentials
5) plot_vcirc
"""
import numpy as np  
     
     
def calculate_acceleration(potgrid,pot):
    """ 
    Calculate acceleration along the grid due to potential pot 
    
    :param potgrid: the grid where the potentials are defined. Can be (R,z) or (x,y,z)
    :param pot:     a 2D or 3D potential in (z,R) or (z,y,x)
    
    :return Acceleration components along the axis, e.g (aR,az) or (ax,ay,az)
    """
    
    if not isinstance(pot,np.ndarray):
        raise ValueError("pot must be a 2D or 3D array")
    
    pshape = pot.shape
    nsize  = potgrid.shape[0]
    if nsize!=len(pshape) :
        raise ValueError("Shape of potential and grid must be the same")
    
    if nsize == 2:
        # 2D case: the grid is in (R,z), so calculate acc_R, acc_z
        R,Z = potgrid
        allOK = pshape[0]==len(Z) and pshape[1]==len(R)
        if not allOK:
            raise ValueError("Axis sizes of potential do not match the grid provided")
        
        dR, dZ = np.diff(R), np.diff(Z) 
        
        # Acceleration components
        ar, az = np.zeros_like(pot), np.zeros_like(pot)
        
        for i in range (len(R)):
            az[:-1,i] = np.diff(pot[:,i])/dZ
            az[-1,i]  = (pot[-1,i]-pot[-2,i])/(Z[-1] - Z[-2])
        
        for k in range (len(Z)):
            ar[k,:-1] = np.diff(pot[k,:])/dR
            ar[k,-1]  = (pot[k,-1]-pot[k,-2])/(R[-1] - R[-2])
        
        acc = (ar,az)
    
    elif nsize == 3:
        # 3D case: the grid is in (x,y,z), so calculate acc_x, acc_y and acc_z
        X,Y,Z = potgrid
        allOK = pshape[0]==len(Z) and pshape[1]==len(Y) and pshape[2]==len(X)
        if not allOK:
            raise ValueError("Axis sizes of potential do not match the grid provided")
        
        dX, dY, dZ = np.diff(X), np.diff(Y), np.diff(Z) 
        
        # Acceleration components
        ax, ay, az = np.zeros_like(pot), np.zeros_like(pot), np.zeros_like(pot)
        
        for i in range (len(X)):
            for j in range (len(Y)):
                az[:-1,j,i] = np.diff(pot[:,j,i])/dZ
                az[-1,j,i]  = (pot[-1,j,i]-pot[-2,j,i])/(Z[-1] - Z[-2])
        
        for i in range (len(X)):
            for k in range (len(Z)):
                ay[k,:-1,i] = np.diff(pot[k,:,i])/dY
                ay[k,-1,i]  = (pot[k,-1,i]-pot[k,-2,i])/(Y[-1] - Y[-2])
       
        for j in range (len(Y)):
            for k in range (len(Z)):
                ax[k,j,:-1] = np.diff(pot[k,j,:])/dX
                ax[k,j,-1]  = (pot[k,j,-1]-pot[k,j,-2])/(X[-1] - X[-2])
        
        acc = (ax,ay,az)
        
    else:
        raise ValueError("Only 2D and 3D potential array are accepted")
        
    return acc


def writeFITS (coordgrid,potentials,npots,names=None,fname="potentials.fits"):
    """ 
    Write a FITS file with n potentials in n extensions 
    
    :param coordgrid: the grid where the potentials are defined. Can be (R,z) or (x,y,z)
    :potentials:      a list of 2D or 3D array of potentials
    :param npots:      number of potentials
    :names:           names of potentials
    :fname:           output FITS file
    """
    try: from astropy.io import fits
    except: raise ModuleError("To write a FITS file you need astropy module")
    
    nsize = coordgrid.shape[0]
    
    nexts = npots
    
    if nsize==2: ctypes = ["R", "z"] 
    elif nsize==3: ctypes = ["x", "y", "z"] 
    else: raise ValueError("Shape of coordgrid must be (2,N) or (3,N)")
    
    head = fits.Header()   
    head['BTYPE'], head['BUNIT'] = 'Potential', '(kpc/Myr)^2'
    for n in range (nsize):
        head['CUNIT%d'%(n+1)] = 'kpc'
        head['CRPIX%d'%(n+1)] = 1
        head['CTYPE%d'%(n+1)] = ctypes[n]
        head['CDELT%d'%(n+1)] = coordgrid[n][1]-coordgrid[n][0]
        head['CRVAL%d'%(n+1)] = coordgrid[n][0]
    
    hdulist = fits.HDUList()
    for i in range (nexts):
        if nexts==1: pot, name = potentials, names
        else: pot, name = potentials[i], names[i]
        hdu = fits.PrimaryHDU(pot)
        hdu.header.extend(head.cards)
        if name: hdu.header["OBJECT"] = name
        hdulist.append(hdu)
    
    hdulist.writeto(fname,overwrite=True)
    print ("Potentials written in %s"%fname)
    return hdulist
    
    
    
def writeHDF5 (coordgrid,potentials,npots,names=None,fname="potentials.h5", flatten=False):
    """ 
    Write a HDF5 file with n potentials in n extensions. Potentials are flattened in a 1D array
    
    :param coordgrid:  the grid where the potentials are defined. Can be (R,z) or (x,y,z)
    :param potentials: a list of 2D or 3D array of potentials
    :param npots:      number of potentials
    :param names:      names of potentials
    :param fname:      output FITS file.
    :param flatten:    if True, potentials are flattened in a 1D array (e.g., of size len(R)*len*(z))
    """
    try: import h5py
    except: raise ModuleError("To write a HDF5 file you need h5py module")
      
    ntabs = npots
    
    # If names are not given, just call them 1-2-3-4
    if names is None:
        names = [str(i+1) for i in range (ntabs)]
        
    # Check whether the arrays are 2D or 3D
    N = coordgrid.shape[0]
    if N==2:
        R, Z = coordgrid
        DIM  = len(R)*len(Z)
        rr,zz = np.meshgrid(R,Z,indexing='ij')
        coords = np.zeros((DIM,N))
        coords[:,0] = np.ravel(rr)
        coords[:,1] = np.ravel(zz)
        sizes = [len(R),len(Z)]
    elif N==3:
        X, Y, Z = coordgrid
        DIM  = len(X)*len(Y)*len(Z)
        xx,yy,zz = np.meshgrid(X,Y,Z,indexing='ij')
        coords = np.zeros((DIM,N))
        coords[:,0] = np.ravel(xx)
        coords[:,1] = np.ravel(yy)
        coords[:,2] = np.ravel(zz)
        sizes = [len(X),len(Y),len(Z)]
    else:
        raise ValueError("Only 2D and 3D potential array are accepted")
    
    # Opening HDF5, create a group and datasets
    f = h5py.File(fname,'w')
    gname = "Potentials"
    grouph = f.create_group("Header")
    group = f.create_group(gname)
    grouph.attrs.create("BoxSizes", sizes, (N,), h5py.h5t.STD_I32LE)
    datasets = []
    
    if flatten:
        # 2 or 3 vectors of size DIM
        datasets.append(f.create_dataset(gname+"/Coordinates", (DIM,N),dtype=h5py.h5t.IEEE_F64LE))
        datasets[0][...] = coords
        for i in range (ntabs):
            # A 1D array representing the potential
            datasets.append(f.create_dataset(gname+"/"+names[i],(DIM,),dtype=h5py.h5t.IEEE_F64LE))
            if ntabs==1: pot = potentials
            else: pot = potentials[i]
            datasets[-1][...] = np.ravel(pot)
     
    else:
        nn = ["/Coord_R", "/Coord_Z"] if N==2 else ["/Coord_X", "/Coord_Y", "/Coord_Z"]
        for i in range (N):
            # 1D vector for each coordinate
            datasets.append(f.create_dataset(gname+nn[i], (sizes[i],),dtype=h5py.h5t.IEEE_F64LE))
            datasets[-1][...] = coordgrid[i]
        for i in range (ntabs):
            # A 2D/3D array representing the potential. Axis order is now (x,y,z)!!
            datasets.append(f.create_dataset(gname+"/"+names[i],sizes,dtype=h5py.h5t.IEEE_F64LE))
            if ntabs==1: pot = potentials
            else: pot = potentials[i]
            datasets[-1][...] = pot.T
            
    f.close()
  
 
    
def plot_potentials(R,Z,potentials,npots,names=None,contours=None,fname="potentials.pdf"):
    """ 
    Plot potentials
    
    :param R,z:        the grid where the potentials are defined
    :param potentials: a list of 2D array of potentials in (z,R)
    :param npots:      number of potentials
    :param names:      names of potentials
    :param countours:  contour levels
    :param fname:      output FITS file. Set to None to just return the object
    """
    try: import matplotlib.pyplot as plt
    except: raise ModuleError("To plot you need matplotlib module")
            
    # Plotting potentials
    fsize = 18
    fig, ax = plt.figure(figsize=(15,10)), []
    nplots = npots
    psize, ncols = 0.4, int(nplots/2)
    if nplots%2==1: ncols+=1
    
    for i in range (ncols):
        ax.append(fig.add_axes([0.07+i*(psize+0.08),0.55,psize,psize]))
    for i in range (nplots-ncols):
        ax.append(fig.add_axes([0.07+i*(psize+0.08),0.55-psize-0.08,psize,psize]))
        
        
    cmap = plt.get_cmap("Greys_r")
    for i in range (len(ax)):
        if nplots==1: pot, name = potentials, names 
        else: pot, name = potentials[i], names[i]
        ax[i].tick_params(labelsize=fsize,which='major',direction='in')
        ax[i].set_xticks(np.arange(R[0], R[-1]+1, 2.))
        ax[i].set_yticks(np.arange(Z[0], Z[-1]+1, 2.))
        ax[i].set_xlabel("Radius (kpc)",fontsize=fsize)
        ax[i].set_ylabel("z (kpc)",fontsize=fsize)
        if name is not None: ax[i].text(0.97,0.9,name,transform=ax[i].transAxes,ha='right',fontsize=fsize)
        ax[i].imshow(potentials[i],extent=[R[0],R[-1],Z[0],Z[-1]],aspect='auto',cmap=cmap)
        if contours is not None: ax[i].contour(R,Z,pot,contours,colors='k')
        else: ax[i].contour(R,Z,pot,colors='k')
        
        ax[i].grid()

    if fname is not None: plt.savefig(fname,bbox_inches='tight')
    
    return fig



def plot_vcirc(R,vcircs,nvcirc,names=None,ltype=None,fname="vcirc.pdf"):
    """ 
    Plot circular velocities
    
    :param R:        Radii
    :param vcircs:   a list of 2D array of potentials in (z,R)
    :param nvcirc:   number of circular velocities
    :param names:    names of velocities
    :param ltype:    line types
    :param fname:    output FITS file. Set to None to just return the object
    """
    try: import matplotlib.pyplot as plt
    except: raise ModuleError("To plot you need matplotlib module")
    
    nplots = len(vcircs)
    
    # Plotting circular velocities
    fig = plt.figure(figsize=(8,6))
    fsize = 18
    plt.tick_params(labelsize=fsize,which='major',direction='in')
    if ltype is None: ltype = np.full(nplots,'-')
    
    for i in range (nplots):
        if nplots==1: vc, name = vcircs, names
        else: vc, name = vcircs[i], names[i]
        if name: plt.plot(R,vcircs[i],ltype[i],label=name)
        else: plt.plot(R,vcircs[i],ltype[i])
        
    plt.xlim(R[0],R[-1])
    plt.xlabel("Radius (kpc)",fontsize=fsize)
    plt.ylabel("Vcirc (km/s)",fontsize=fsize)
    plt.grid()
    plt.legend()
    if fname is not None: plt.savefig(fname,bbox_inches='tight')
    return fig
    
    