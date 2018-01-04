from discH.src.pot_disc import test_poly, exponetial_discpy
import numpy as np
import matplotlib.pyplot as plt

'''
#Poly flare
coef=np.array([1.,2.,3.])

R=np.linspace(0,10,50)

y=[]

for r in R:
    y.append(test_poly(r,coef))

y=np.array(y)

plt.plot(R,y)
plt.show()
'''

#Exp disc
Rd=2.
sigma0=10.

R=np.linspace(0,10,50)

y=[]

for r in R:
    y.append(exponetial_discpy(r,sigma0=sigma0,Rd=Rd))

y=np.array(y)

plt.plot(R,y)
plt.show()