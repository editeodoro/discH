from test import interpy, interpupy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import time
from scipy.interpolate import dfitpack

xa=np.array([0,3,5,10,12,20,25,30,32])
ya=np.array([2,3,6,8,9,14,18,19,25])
x=np.linspace(0,35,10000)

plt.scatter(xa,ya,s=30)

t1=time.time()
ret=interpy(x,xa,ya)
print('Cython', time.time()-t1)

plt.plot(ret[:,0],ret[:,1])

t1=time.time()
f=UnivariateSpline(xa,ya,s=0,k=2)
y=f(x)
print('Python', time.time()-t1)

plt.plot(x,y)


#plt.show()

print(interpupy(xa,ya,k=1,s=0.0))

#data = dfitpack.fpcurf0(xa,ya,k=2,w=None,xb=None,xe=None,s=0)
#data=np.array(data)
#for i in range(len(data)):
#    print(i,data[i])

import discH.src.pot_halo.pot_c_ext.isothermal_halo as mod

fintegrand = LowLevelCallable.from_cython(mod, 'integrand_hiso')