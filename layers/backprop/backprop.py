import numpy as np


def fullyconnected_layer(weights, cost, dactivations, previousactivations):
	gbias = np.multiply(cost, dactivations)
	gweights = np.dot(gbias, np.transpose(previousactivations))
	newcost = np.dot(np.transpose(np.multiply(weights, dactivations)), cost)
	return (gbias,gweights,newcost)

def calcgradients(type, weights, cost, dactivations, previousactivations):
	if (type == "fullyconnected"):
		return fullyconnected_layer(weights=weights, cost=cost, dactivations=dactivations, previousactivations=previousactivations)
	else:
		raise Exception("calcgradients failed activation type undefined")