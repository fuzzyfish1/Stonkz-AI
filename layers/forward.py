import numpy as np

def fullyconnected_layer(weights, bias, previousactivations):
	return np.add(np.dot(weights, previousactivations), bias)

def feedforward(type,weights,bias,previousactivations):
	if (type == "fullyconnected"):
		return fullyconnected_layer(weights= weights,bias = bias,previousactivations = previousactivations)
	else:
		raise ("!!!layertype not found big fuck time!!!")