import numpy as np
import handyfuncs.listfuncs as lf

def dsigmoid(x):
	return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

def didentity(x):
	return np.array(lf.fill2D(num=1, rows=len(x), cols=1))

def dReLU(x):
	j = x
	j[j <= 0] = 0
	j[j > 0] = 1
	return j

def dleaky_ReLU(x):
	j = x
	j[j <= 0] = .01
	j[j > 0] = 1
	return j

def dactivate(type, z):
	if (type == "sigmoid"):
		return dsigmoid(z)
	
	elif (type == "ReLU"):
		return dReLU(z)
	
	elif (type == "identity"):
		return didentity(z)
	
	elif (type == "leaky ReLU"):
		return dleaky_ReLU(z)
	
	else:
		raise Exception("!!!the dactivation does not exist !!! fuck")
	