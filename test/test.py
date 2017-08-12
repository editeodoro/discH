import discH
import time
import numpy as np
from discH.src.pot_halo import potential_iso
from discH.src.pot_halo.pot_halo import isothermal_halo
#print(discH.potentialh_iso2(R=10,Z=2,d0=1,rc=0.5,e=0,rcut=20,toll=1e-4))


def main():
    d0=3
    rc=2
    e=0.
    mcut=100


    R=np.linspace(0,15,100000)
    Z=np.linspace(0,5,100000)

#t1=time.time()
#b=a.potential(R,Z,1e-2)
#print('res',b)
#print('-2',time.time()-t1)

    a=discH.isothermal_halo(d0,rc,e,mcut)
    t1=time.time()
    b=a.potential(R,Z,grid=False, toll=1e-4,nproc=1)
    print(b[0])
    print('n=1',time.time()-t1)

    a=discH.isothermal_halo(d0,rc,e,mcut)
    t1=time.time()
    b=a.potential(R,Z,grid=False, toll=1e-4,nproc=2)
    print(b[0])
    print('n=2',time.time()-t1)

    a=discH.NFW_halo(d0,rc,e,mcut)
    t1=time.time()
    b=a.potential(R,Z,grid=False, toll=1e-4,nproc=1)
    print(b[0])
    print('n=1',time.time()-t1)

    a=discH.NFW_halo(d0,rc,e,mcut)
    t1=time.time()
    b=a.potential(R,Z,grid=False, toll=1e-4,nproc=2)
    print(b[0])
    print('n=2',time.time()-t1)

    #def potential_parallel(self,R,Z,grid=False, toll=1e-4, nproc=2):

'''

#t1=time.time()
#b=a.potential2(R,Z,1e-4)
#print('res',b)
#print('-4 (2)',time.time()-t1)

#t1=time.time()
#b=a.potential(R,Z,1e-8)
#print('res',b)
#print('-8',time.time()-t1)

print('#################')

#(double[:] rtab,double[:] ztab, double d0,double rc,double e, double rcut, hlaw ,toll=1E-4)


#t1=time.time()
#b=np.empty(len(R))
#for i in range(len(R)):
#    b[i]=discH.potentialh_iso(R[i],Z[i],d0,rc,e, mcut/np.sqrt(2), 1e-4)
#print('-4 old bad',time.time()-t1)

#t1=time.time()
#b=discH.cpot_halo(R, Z, d0,rc,e, mcut/np.sqrt(2), 'iso' ,toll=1E-4)
#print(b[0])
#print('-4 old good',time.time()-t1)




t1=time.time()
b=potential_iso(R,Z,d0=d0,rc=rc,e=e,mcut=mcut,toll=1e-4,grid=True)
print(b[0])
print(b.shape)
#print('res',b)
print('-4',time.time()-t1)

t1=time.time()
a=discH.isothermal_halo(d0,rc,e,mcut)
b=a.potential(R,Z,grid=True,toll=1e-4)
print(b[0])
print(b.shape)
print('-4',time.time()-t1)
'''

if __name__ == '__main__':
    main()
    #import cProfile
    #cProfile.run('main()',sort='time')
