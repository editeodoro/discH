from test import *
from scipy.special import ellipe, ellipk
import time

p=0.9


print('E python scipy')
t1=time.time()
a=ellipe(p*p)
t2=time.time()-t1
print('Res, Time',a, t2)

print('E cython scipy')
t1=time.time()
a=ellipe_sc(p*p)
t2=time.time()-t1
print('Res, Time',a, t2)

print('E cython gsl')
t1=time.time()
a=ellipe_gs(p)
t2=time.time()-t1
print('Res, Time',a, t2)


print('K python scipy')
t1=time.time()
a=ellipk(p*p)
t2=time.time()-t1
print('Res, Time',a, t2)

print('K cython scipy')
t1=time.time()
a=ellipk_sc(p*p)
t2=time.time()-t1
print('Res, Time',a, t2)

print('K cython gsl')
t1=time.time()
a=ellipk_gs(p)
t2=time.time()-t1
print('Res, Time',a, t2)
