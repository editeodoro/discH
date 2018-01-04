from test import func, funcsc, hy
import scipy.special as sf
from math import sqrt
import time
from scipy.special import hyp2f1


x=0.6
t1=time.time()
r=func(sqrt(x))
tempo=time.time()-t1
print('GSL',r,'in',tempo,'s')


t1=time.time()
r=funcsc(x)
tempo=time.time()-t1
print('Scipy',r,'in',tempo,'s')

t1=time.time()
alfa=1
beta=3
a=3-alfa
b=beta-alfa
c=4-alfa
x=-2
print(a,b,c)
r= hyp2f1(a,b,c,x)
print(r)
print('Scipy',time.time()-t1)

t1=time.time()
alfa=1
beta=3
a=3-alfa
b=beta-alfa
c=4-alfa
x=-2
print(a,b,c)
r= hy(a,b,c,x)
print(r)
print('Cy',time.time()-t1)


alfa=1
beta=3
a=3-alfa
b=beta-alfa
c=4-alfa
rs=1
m=5
x=m/rs
r=hy(a,b,c,-x)
print(r)
r2=-r*((x)**(-alfa))*m*m*m/(alfa-3)
print(r2)
#NB in GSL ellipk(x) è equivalente a la funzione di scipy ellipk(x*x), per cui
#se voglio usare GSL come usavo prima ellipk devo scrive ellipk(sqrt(x))




