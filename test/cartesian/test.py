import numpy as np

def cartesian(*arrays):
	"""
	Make a cartesian combined arrays from different arrays e.g.
	al=np.linspace(0.2,5,50)
	ql=np.linspace(0.1,2,50)
	par=cartesian(al,ql)
	:param arrays:
	:return:
	"""
	mesh = np.meshgrid(*arrays)  # standard numpy meshgrid
	dim = len(mesh)  # number of dimensions
	elements = mesh[0].size  # number of elements, any index will do
	flat = np.concatenate(mesh).ravel()  # flatten the whole meshgrid
	reshape = np.reshape(flat, (dim, elements)).T  # reshape and transpose
	return reshape

R=np.linspace(0.1,10,5)
Z=np.linspace(1,3,3)

b=cartesian(R,Z)

print(b)