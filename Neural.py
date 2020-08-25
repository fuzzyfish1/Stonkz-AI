import listfuncs as ls
import numpy as np

class AI:

	def __init__(self, configfile):

		self.configpath = configfile
		self.config = ls.deserialize(filepath= configfile)

		self.totalcost = self.config["totalcost"]
		self.iterations = self.config["iterations"]
		self.savespace = self.config["save-directory"]
		self.architecture = self.config["architecture"]
		self.width = len(self.architecture)
		#print("savespace is "+ self.savespace)

		halfserializedweights = ls.deserialize(filepath=self.savespace+"weights.json")

		self.weights = halfserializedweights

		for w in range(len(halfserializedweights)):
			self.weights[w] = np.array(halfserializedweights[w])

		halfserializedbiases = ls.deserialize(filepath=self.savespace + "biases.json")

		self.bias = halfserializedbiases

		for b in range(len(halfserializedbiases)):
			self.bias[b] = np.array(halfserializedbiases[b])
		self.temphist= []
		self.tempiter =[]
		#print("i beleive the network init successfully")

	def save(self):

		# turning everything into a list in a list in a list to save to 1 file

		saveweights = self.weights

		for x in range(len(self.weights)):

			if( type(saveweights[x]) is int):
				saveweights[x] = self.weights[x]
			else:
				saveweights[x] = self.weights[x].tolist()

		savebiases = self.bias

		for x in range(len(self.bias)):

			if( type(savebiases[x]) is int):
				savebiases[x] = self.bias[x]
			else:
				savebiases[x] = self.bias[x].tolist()


		ls.serialize(filepath = self.savespace +"weights.json", obj=saveweights)
		ls.serialize(filepath = self.savespace +"biases.json", obj=savebiases)

		config = {
			"save-directory" : self.savespace,
			"iterations" : self.iterations,
			"totalcost": self.totalcost,
			"architecture": self.architecture
		}

		ls.serialize( filepath = self.configpath, obj = config)

		if(config["iterations"] % 1000):
			ls.backup(filepath_of_backup=self.savespace +"weights.json", filepath= self.savespace + "weightsbackup.json")
			ls.backup(filepath_of_backup=self.savespace + "biases.json", filepath= self.savespace + "biasbackup.json")
			ls.backup(filepath_of_backup=self.configpath , filepath=self.savespace + "confbackup.json")

	def printweights(self):

		print("\n"+"BEHOLD MY print statements"+"\n\n")

		print("architecture"+"\n")
		print(self.architecture)

		print("\n\n"+"weights"+"\n")
		for w in self.weights:
			print(w)

		print("\n\n"+"biases"+"\n")
		for b in self.bias:
			print(b)

		print("\n"+"width: "+str(self.width))

	def predict(self, data):

		# initializing the net

		perceptrons = np.array(data)
		perceptrons = (1/(1+np.exp(-perceptrons))) - 0.5

		#3rows 1 cols

		self.activations = [0 for e in range(self.width)]
		self.z = [0 for e in range(self.width)]
		self.activations[0] = perceptrons

		# x starts from 1
		# x = 0 is perceptrons
		# x is the activation being calculated

		#print("\n" + "self.activations: " + str(0) + "\n")
		#print(self.activations[0])
		# checking if inputs are vectorized
		if(not ls.isvector(self.activations[0])):
			print("!!!!!you got a really big fucking error!!!!!!!!-----------------!!!!!!!------------------!!!!!!--------------------------------!!!!!!------------------------!!!!!!-------------!!!!!")
			print("inputs are not vectors")

		# checking if size of inputs matches architecture
		if (self.architecture[0][2] != len(data)):
			print("!!!!!you got a really big fucking error!!!!!!!!-----------------!!!!!!!------------------!!!!!!--------------------------------!!!!!!------------------------!!!!!!-------------!!!!!")
			print("the input data vector does not match architecture")

		# feed forward algorithm begins
		for x in range(1,self.width):
			#print("x: "+str(x))
			#print("calculating:")
			#print(x)

			# <<<<<<<<<<<<<<<<<<<<<---------------------    z   calculations     ------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>

			if(self.architecture[x][0] == "fullyconnected"):
				self.z[x] = np.add(np.dot(self.weights[x], self.activations[x-1]), self.bias[x])
				#print(type(self.z[x]))

			# <<<<<<<<<<<<<<<<<<<<--------------------     activation calculations --------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>
			if(self.architecture[x][1] == "sigmoid"):
				self.activations[x] = 1/(1+np.exp(-self.z[x]))

			if(self.architecture[x][1] == "ReLU"):
				self.activations[x] = np.maximum(self.z[x], 0.0)
				#print(type(self.activations[x]))

			if(self.architecture[x][1] == "identity"):
				self.activations[x] = self.z[x]

			'''
			if(self.architecture[x][1] == "binary step function"):

			if(self.architecture[x][1] == "ReLU"):
			if(self.architecture[x][1] == "ReLU"):
			if(self.architecture[x][1] == "ReLU"):
			'''

			#print("activations:")
			#print(x)
			#print(self.activations[x-1])
			#print(self.weights[x])

		self.answer = self.activations[(self.width -1)].tolist()

		#print("\n"+"self.answer"+"\n")
		#print(type(self.answer))

		return self.answer

	def backpropagate(self, input="none", output=0):

		dactivations = [0 for x in range(self.width)]
		gweights = [0 for x in range(self.width)]
		gbias = [0 for x in range(self.width)]

		def printgradients():
			for f in gweights:
				print(f)
			for b in gbias:
				print(b)

		def drelu(x):
			x[x <= 0] = 0
			x[x > 0] = 1
			return x

		#print("perceptrons")
		#print(self.activations[0])

		if(input == "none"):
			# assume the neural nets last answers
			#print("\n\n\n\n"+"noooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooone"+"\n\n\n\n")
			answer = np.array(self.answer)
		else:
			answer = np.array(self.predict(input))

		correct = np.array(output)

		#print("\n"+"correct"+"\n")
		#print(correct)

		self.cost = (answer - correct)**2

		self.avgcost = np.sum(self.cost)/ len(self.cost)


		C = [0 for x in range(self.width)]
		#print(len(C))
		C[-1] = 2*(answer-correct)

		#C = del C/del cost

		#behold my fucking math



		for x in range(1,self.width):
			l = self.width -x

	#		print("\n"+"new prop")
	#		print(l)
	#		print(self.architecture[l])

			if (self.architecture[l][1] == "sigmoid"):
				dactivations[l] = np.exp(-self.z[l]) / ((1 + np.exp(-self.z[l])) ** 2)

			elif (self.architecture[l][1] == "ReLU"):
				dactivations[l] = drelu(self.z[l])

			elif(self.architecture[l][1] == "identity"):
				dactivations[l] = ls.fill2D(num=1, rows=len(self.z[l]), cols=1)

			if (self.architecture[x][0] == "fullyconnected"):
				gbias[l]=np.multiply(C[l], dactivations[l])
				#print("\n"+"gbias: "+str(l)+"\n")
				#print(C[l])
				C[l-1] = np.dot(np.transpose(np.multiply(self.weights[l], dactivations[l])), C[l])
				gweights[l] = np.dot(gbias[l], np.transpose(self.activations[l-1]))

			self.weights[l] = np.add(self.weights[l], -gweights[l])
			self.bias[l] = np.add(self.bias[l], -gbias[l])

		self.iterations += 1
		self.totalcost += self.avgcost

		#printgradients()

		#self.temphist.append(self.totalcost/self.iterations)
		#self.tempiter.append(self.iterations)
		#plt.plot(self.tempiter,self.temphist)

		self.save()

class AI_initer:

	@staticmethod
	def buildconfig():

		# make sure you have created a file named

	#	print("\n\n"+"welcome to build config"+"\n\n")

		savespot = input("input config file name in same directory"+"\n")
		savedir = input("input save directory"+"\n")

		savestring = savedir + "/"

		config = {

			"save-directory" : savestring,
			"iterations" : 0,
			"totalcost": 0,
			"architecture":
			[
				[
					"fullyconnected", "sigmoid", 14
					# perceptrons the first and second thing shouldnt matter
				],
				[
					"fullyconnected", "ReLU", 24
				],
				[
					"fullyconnected", "ReLU", 18
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

			weights[x] = ls.randomize2D(rows=config["architecture"][x][2],cols=config["architecture"][x-1][2])
			bias[x] = ls.randomize2D(rows=config["architecture"][x][2],cols=1)


		ls.serialize(filepath=savestring + "weights.json", obj=weights)
		ls.serialize(filepath=savestring + "biases.json", obj=bias)
