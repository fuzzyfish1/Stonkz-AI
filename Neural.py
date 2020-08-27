import listfuncs as ls
import numpy as np


class AI:
	
	def __init__(self, configfile):
		
		self.configpath = configfile
		self.config = ls.deserialize(filepath=configfile)
		self.totalcost = self.config["totalcost"]
		self.iterations = self.config["iterations"]
		self.savespace = self.config["save-directory"]
		self.architecture = self.config["architecture"]
		self.width = len(self.architecture)
		self.activations = [0 for e in range(self.width)]
		self.z = [0 for e in range(self.width)]
		
		halfserializedweights = ls.deserialize(filepath=self.savespace + "weights.json")
		
		self.weights = halfserializedweights
		
		for w in range(len(halfserializedweights)):
			self.weights[w] = np.array(halfserializedweights[w])
		
		halfserializedbiases = ls.deserialize(filepath=self.savespace + "biases.json")
		
		self.bias = halfserializedbiases
		
		for b in range(len(halfserializedbiases)):
			self.bias[b] = np.array(halfserializedbiases[b])
	
	def save(self):
		
		# turning everything into a list in a list in a list to save to 1 file
		
		saveweights = self.weights
		
		for x in range(len(self.weights)):
			
			if (type(saveweights[x]) is int):
				saveweights[x] = self.weights[x]
			else:
				saveweights[x] = self.weights[x].tolist()
		
		savebiases = self.bias
		
		for x in range(len(self.bias)):
			
			if (type(savebiases[x]) is int):
				savebiases[x] = self.bias[x]
			else:
				savebiases[x] = self.bias[x].tolist()
		
		ls.serialize(filepath=self.savespace + "weights.json", obj=saveweights)
		ls.serialize(filepath=self.savespace + "biases.json", obj=savebiases)
		
		config = {
			"save-directory": self.savespace,
			"iterations": self.iterations,
			"totalcost": self.totalcost,
			"architecture": self.architecture
		}
		
		ls.serialize(filepath=self.configpath, obj=config)
		
		if (config["iterations"] % 1000):
			ls.backup(filepath_of_backup=self.savespace + "weights.json",
			          filepath=self.savespace + "weightsbackup.json")
			ls.backup(filepath_of_backup=self.savespace + "biases.json",
			          filepath=self.savespace + "biasbackup.json")
			ls.backup(filepath_of_backup=self.configpath, filepath=self.savespace + "confbackup.json")
	
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
		
		def activate(type, toactivate):
			if (type == "sigmoid"):
				return 1 / (1 + np.exp(-toactivate))
			
			if (type == "ReLU"):
				return np.maximum(toactivate, 0.0)
			
			if (type == "identity"):
				return toactivate
		
		self.activations[0] = activate(type=self.architecture[0][1], toactivate=np.array(data))
		
		# x starts from 1
		# x = 0 is perceptrons
		# x is the activation being calculated
		
		# checking if inputs are vectorized
		
		# checking if inputs match architecture
		if (not ls.isvector(self.activations[0])):
			print(
				"!!!!!you got a really big fucking error!!!!!!!!-----------------!!!!!!!------------------!!!!!!--------------------------------!!!!!!------------------------!!!!!!-------------!!!!!")
			print("inputs are not vectors")
		
		if (self.architecture[0][2] != len(data)):
			print(
				"!!!!!you got a really big fucking error!!!!!!!!-----------------!!!!!!!------------------!!!!!!--------------------------------!!!!!!------------------------!!!!!!-------------!!!!!")
			print("the input data vector does not match architecture")
		
		# feed forward algorithm begins
		for x in range(1, self.width):
			
			# <<<<<<<<<<<<<<<<<<<<<---------------------    z   calculations     ------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>
			
			if (self.architecture[x][0] == "fullyconnected"):
				self.z[x] = np.add(np.dot(self.weights[x], self.activations[x - 1]), self.bias[x])
			
			self.activations[x] = activate(type=self.architecture[x][1], toactivate=self.z[x])
			
			'''
			if(self.architecture[x][1] == "binary step function"):
			'''
		
		self.answer = self.activations[(self.width - 1)].tolist()
		
		return self.answer
	
	def backpropagate(self, input="none", output=0):
		
		dactivations = [0 for x in range(self.width)]
		gweights = [0 for x in range(self.width)]
		gbias = [0 for x in range(self.width)]
		
		def printgradients():
			for w in range(len(gweights)):
				print("\n\n" + "gweights: "+str(w) + "\n\n")
				print(gbias[w])
			for b in range(len(gbias)):
				print("\n\n" + "gbiases: "+str(b) + "\n\n")
				print(gbias[b])
		
		def drelu(x):
			x[x <= 0] = 0
			x[x > 0] = 1
			return x
		
		# print("perceptrons")
		# print(self.activations[0])
		
		if input == "none":
			# assume the neural nets last answers
			answer = np.array(self.answer)
		else:
			answer = np.array(self.predict(input))
		
		correct = np.array(output)
		
		self.cost = (correct - answer) ** 2
		
		self.avgcost = np.sum(self.cost) / len(self.cost)
		
		C = [0 for x in range(self.width)]
		C[-1] =  (correct - answer)
		#print("\n\n"+"Cost"+"\n\n")
		#print(C[-1])
		
		# C = del C/del cost
		
		# behold my fucking math
		
		for x in range(1, self.width):
			l = self.width - x
			
			if (self.architecture[l][1] == "sigmoid"):
				dactivations[l] = np.exp(-self.z[l]) / ((1 + np.exp(-self.z[l])) ** 2)
			
			elif (self.architecture[l][1] == "ReLU"):
				dactivations[l] = drelu(self.z[l])
			
			elif (self.architecture[l][1] == "identity"):
				dactivations[l] = np.array(ls.fill2D(num=1, rows=len(self.z[l]), cols=1))
			
			if (self.architecture[x][0] == "fullyconnected"):
				#print("\n"+"dactivations: "+str(l)+"\n")
				#print(dactivations[l])
				gbias[l] = np.multiply(C[l], dactivations[l])
				C[l - 1] = np.dot(np.transpose(np.multiply(self.weights[l], dactivations[l])), C[l])
				gweights[l] = np.dot(gbias[l], np.transpose(self.activations[l - 1]))
		
		for x in range(1, self.width):
			self.weights[x] = np.add(self.weights[x], -gweights[x])
			self.bias[x] = np.add(self.bias[x], -gbias[x])
		#printgradients()
		
		self.iterations += 1
		self.totalcost += self.avgcost
		
		# printgradients()
		
		# self.temphist.append(self.totalcost/self.iterations)
		# self.tempiter.append(self.iterations)
		# plt.plot(self.tempiter,self.temphist)
		
		self.save()


class AI_initer:
	
	@staticmethod
	def buildconfig():
		savespot = input("input config file name in same directory" + "\n")
		savedir = input("input save directory" + "\n")
		
		savestring = savedir + "/"
		
		config = {
			
			"save-directory": savestring,
			"iterations": 0,
			"totalcost": 0,
			"architecture":
				[
					[
						"fullyconnected", "sigmoid", 14
						# perceptrons the first and second thing shouldnt matter
					],
					[
						"fullyconnected", "sigmoid", 24
					],
					[
						"fullyconnected", "sigmoid", 18
					],
					[
						"fullyconnected", "sigmoid", 9
					],
					[
						"fullyconnected", "sigmoid", 4
					]
				]
		}
		
		ls.serialize(obj=config, filepath=savespot)
		
		# randomizing the neural network
		
		weights = [0 for x in range(len(config["architecture"]))]
		bias = [0 for x in range(len(config["architecture"]))]
		
		for x in range(1, len(config["architecture"])):
			weights[x] = ls.fill2D(num =.1,rows=config["architecture"][x][2], cols=config["architecture"][x - 1][2])
			bias[x] = ls.fill2D(num =.1,rows=config["architecture"][x][2], cols=1)
		
		ls.serialize(filepath=savestring + "weights.json", obj=weights)
		ls.serialize(filepath=savestring + "biases.json", obj=bias)
