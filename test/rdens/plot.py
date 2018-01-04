import matplotlib.pyplot as plt
import numpy as np
from discH.src.pot_disc import poly_flarew, constantw, asinh_flarew, tanh_flarew, poly_exponentialw, gaussianw, fratlaww

R=np.linspace(0,30,100)

sigma0=10
Rlimit=15
Rd=5
Rd2=25
R0=1
coeff=[2,0.005,0.01]

'''
y=poly_exponentialw(R,Rd,coeff)
plt.plot(R,y)

y=gaussianw(R,Rd,R0)
plt.plot(R,y)

y=fratlaww(R,Rd,Rd2,2)
plt.plot(R,y)
'''

y=poly_flarew(R,coeff,Rlimit=Rlimit)
plt.plot(R,y)

y=asinh_flarew(R,h0=2,c=1, Rf=Rd2, Rlimit=None)
plt.plot(R,y)

y=tanh_flarew(R,h0=2,c=1, Rf=Rd2, Rlimit=None)
plt.plot(R,y)

plt.show()