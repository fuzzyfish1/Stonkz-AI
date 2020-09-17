from handyfuncs import listfuncs as lf
from handyfuncs import savefuncs as sf
from activations import activations as act
from activations import dactivations as dact
from layers import forward as layer
from layers.backprop import backprop as bprop

import numpy as np

class AI:
	
	def __init__(self, configfile):
		
		self.configpath = configfile
		self.config = sf.deserialize(filepath=configfile)
		self.totalcost = self.config["totalcost"]
		self.iterations = self.config["iterations"]
		self.savespace = self.config["save-directory"]
		self.architecture = self.config["architecture"]
		self.width = len(self.architecture)
		self.activations = [0 for e in range(self.width)]
		self.z = self.activations
		self.learningrate = 1
		
		
		self.weights = sf.loadlistofarrays(filepath=self.savespace + "weights.json")
		self.bias = sf.loadlistofarrays(filepath=self.savespace + "biases.json")
	
	def save(self):
		
		sf.savelistofarrays(filepath=self.savespace + "weights.json", obj=self.weights)
		sf.savelistofarrays(filepath=self.savespace + "biases.json", obj=self.bias)
		
		config = {
			"save-directory": self.savespace,
			"iterations": self.iterations,
			"totalcost": self.totalcost,
			"architecture": self.architecture
		}
		
		sf.serialize(filepath=self.configpath, obj=config)
		
		if (config["iterations"] % 1000):
			sf.backup(filepath_of_backup=self.savespace + "weights.json",
			          filepath=self.savespace + "weightsbackup.json")
			sf.backup(filepath_of_backup=self.savespace + "biases.json",
			          filepath=self.savespace + "biasbackup.json")
			sf.backup(filepath_of_backup=self.configpath, filepath=self.savespace + "confbackup.json")
	
	def printweights(self):
		
		print("\n" + "BEHOLD MY print statements" + "\n\n")
		
		print("architecture" + "\n")
		print(self.architecture)
		
		print("\n\n" + "weights" + "\n")
		for w in self.weights:
			print(w)
		
		print("\n\n" + "biases" + "\n")
		for b in self.bias:
			print(b)
		
		print("\n" + "width: " + str(self.width))
	
	def predict(self, data):
		
		self.activations[0] = act.activate(type=self.architecture[0][1], z=np.array(data))
		
		# x starts from 1
		# x = 0 is perceptrons
		# x is the activation being calculated
		
		# checking if inputs are vectorized
		# checking if inputs match architecture
		if (not lf.isvector(self.activations[0])):
			print("input data: ")
			print(self.activations[0])
			raise Exception("input data is not a vector")
		
		elif (self.architecture[0][2] != len(data)):
		
			print("!!!!!you got a really big fucking error!!!!!!!!-----------------!!!!!!-------------!!!!!")
			raise Exception("the input data does not match architecture")
		# feed forward algorithm begins
		
		for x in range(1, self.width):
			self.z[x] = layer.feedforward(type = self.architecture[x][0],weights=self.weights[x],bias=self.bias[x],previousactivations=self.activations[x - 1])
			self.activations[x] = act.activate(type=self.architecture[x][1], z=self.z[x])
		
		self.answer = self.activations[-1]
		'''
		if(np.isnan(self.answer)):
			raise Exception("inf shit")
		elif (np.isinf(self.answer)):
			raise Exception("nan shit")
		'''
		return self.answer.tolist()
	
	def backpropagate(self, input="none", output=0):
		
		dactivations = [0 for x in range(self.width)]
		gweights = [0 for x in range(self.width)]
		gbias = [0 for x in range(self.width)]
		C = [0 for x in range(self.width)]
		
		def printgradients():
			for w in range(len(gweights)):
				print("\n\n" + "gweights: "+str(w) + "\n\n")
				print(gbias[w])
			for b in range(len(gbias)):
				print("\n\n" + "gbiases: "+str(b) + "\n\n")
				print(gbias[b])
		
		if input == "none":
			# assume the neural nets last answers
			answer = np.array(self.answer)
		else:
			answer = np.array(self.predict(input))
		
		correct = np.array(output)
		self.cost = (correct-answer) ** 2
		self.avgcost = np.sum(self.cost) / len(self.cost)
		
		C[-1] = 2*(correct-answer)
		
		print("first cost")
		print(C[-1])
		print("answer is")
		print(self.answer)
		print("real is")
		print(correct)
		
		# C = del C/del cost
		
		# behold my fucking math
		
		for x in range(1, self.width):
			# l is the current layer in question that is being propagated
			l = self.width - x
			
			dactivations[l] = dact.dactivate(type=self.architecture[l][1], z=self.z[l])
			things = bprop.calcgradients(type= self.architecture[l][0],cost = C[l],weights=self.weights[l],previousactivations= self.activations[l-1],dactivations=dactivations[l])
			
			gbias[l] = things[0]
			gweights[l] = things[1]
			C[l-1] = things[2]

		# printgradients()
		for x in range(1, self.width):
			self.weights[x] = np.add(self.weights[x], (lf.capfunc(x= gweights[x],uppercap= self.learningrate, lowercap=-self.learningrate) * -1))
			self.bias[x] = np.add(self.bias[x], (lf.capfunc(x= gbias[x], uppercap=self.learningrate, lowercap= -self.learningrate) * -1))
		
		self.iterations += 1
		self.totalcost += self.avgcost
		
		self.save()
