import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def ReLU(x):
	return np.maximum(x, 0.0)

def identity(x):
	return x

def leaky_ReLU(x):
	return np.where(x > 0, x, x * 0.01)

def activate(type, z):
	if (type == "sigmoid"):
		return sigmoid(z)
	
	elif (type == "ReLU"):
		return ReLU(z)
	
	elif (type == "identity"):
		return identity(z)
	
	elif (type == "leaky ReLU"):
		return leaky_ReLU(z)
	
	else:
		raise Exception("!!!the activation does not exist !!! fuck")
	
