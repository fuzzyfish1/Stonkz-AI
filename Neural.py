import listfuncs as ls
import numpy as np

class AI:

	def squash(self, inputs):
		return 1/(1+np.exp(-inputs))

	def __init__(self, configfile):

		self.config = ls.deserialize(filepath= configfile)

		self.iterations = self.config["iterations"]
		self.savespace = self.config["save-directory"]
		self.architecture = self.config["architecture"]
		self.width= len(self.architecture)
		#print("savespace is "+ self.savespace)

		halfserializedweights = ls.deserialize(filepath=self.savespace+"weights.json")

		self.weights = halfserializedweights

		for w in range(len(halfserializedweights)):
			self.weights[w] = np.array(halfserializedweights[w])

		halfserializedbiases = ls.deserialize(filepath=self.savespace + "biases.json")

		self.bias = halfserializedbiases

		for b in range(len(halfserializedbiases)):
			self.bias[b] = np.array(halfserializedbiases[b])

		#print("i beleive the network init successfully")

	def save(self):

		# turning everything into a list in a list in a list to save to 1 file

		saveweights = self.weights

		for x in range(len(self.weights)):
			#print(x)
			saveweights[x] = self.weights[x].tolist()

		savebiases = self.bias

		for b in range(len(self.bias)):
			savebiases[b] = self.bias[b].tolist()

		ls.serialize(filepath = self.savespace +"weights.json", obj=saveweights)
		ls.serialize(filepath = self.savespace +"biases.json", obj=savebiases)

		config = {
			"save-directory" : self.savespace,
			"iterations" : self.iterations,
			"architecture": self.architecture
		}

		ls.serialize( filepath = self.configpath, obj = config)

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

		perceptrons = self.squash(np.array(data))    #3rows 1 cols

		self.activations = [0 for e in range(self.width)]
		self.z = [0 for e in range(self.width)]
		self.activations[0] = perceptrons

		# x starts from 1
		# x = 0 is perceptrons
		# x is the activation being calculated
		for x in range(1,self.width):
			#print("calculating:")
			#print(x)

			if(self.architecture[0][2] != len(self.activations[0])):
				print("!!!!!you got a really big fucking error!!!!!!!!-----------------!!!!!!!------------------!!!!!!--------------------------------!!!!!!------------------------!!!!!!-------------!!!!!")
				print("the input data vector does not match architecture")

			if(self.architecture[x][0] == "fullyconnected"):
				self.z[x] = np.add(np.dot(self.weights[x], self.activations[x-1]), self.bias[x])

			if(self.architecture[x][1] == "sigmoid"):

				self.activations[x] = self.squash(self.z[x])

			#print("activations:")
			#print(x)
			#print(self.activations[x-1])
			#print(self.weights[x])

		answer = self.activations[-1]

		return answer

	def backpropagate(self,input , output):

		ins = np.array(input)

		def dsigmoid(z):
			return np.exp(-z)/((1+np.exp(-z))**2)

		answer = np.array(self.predict(input))
		correct = np.array(output)

		cost = (answer - correct)**2

		C = [0 for x in range(self.width)]
		print(len(C))
		C[-1] = 2*(answer-correct)

		#C = del C/del cost

		#behold my fucking math

		dsig = [0 for x in range(self.width)]
		gweights = [0 for x in range(self.width)]
		gbias = [0 for x in range(self.width)]

		for x in range(1,self.width):
			l = self.width -x

			print("\n"+"new prop")
			print(l)
			print(self.architecture[l])

			if (self.architecture[l][1] == "sigmoid"):
				dsig[l] = dsigmoid(self.z[l])

			if (self.architecture[x][0] == "fullyconnected"):
				gbias[l]=np.multiply(C[l], dsig[l])
				C[l-1] = np.dot(np.transpose(np.multiply(self.weights[l], dsig[l])), C[l])
				gweights[l] = np.dot(gbias[l], np.transpose(self.activations[l-1]))






			self.weights[l] = self.weights[l] - gweights[l]
			self.bias[l] = self.bias[l] - gbias

		self.iterations += 1

class AI_initer:

	@staticmethod
	def buildconfig():

		# make sure you have created a file named

		print("\n\n"+"welcome to build config"+"\n\n")

		savespot = input("input config file name in same directory"+"\n")
		savedir = input("input save directory"+"\n")

		savestring = savedir + "/"

		config = {

			"save-directory" : savestring,
			"iterations" : 0,
			"architecture":
			[
				[
					"fullyconnected", "sigmoid", 2
					# perceptrons the first and second thing shouldnt matter
				],
				[
					"fullyconnected", "sigmoid", 4
				],
				[
					"fullyconnected", "sigmoid", 3
				]
			]
		}

		ls.serialize(obj=config, filepath=savespot)

		# randomizing the neural network

		weights = [0 for x in range(len(config["architecture"]))]
		bias = [0 for x in range(len(config["architecture"]))]
		print("\n")
		for x in range(1, len(config["architecture"])):
			print("weights["+str(x)+"]"+"\n")
			weights[x] = ls.randomize2D(rows= config["architecture"][x][2],cols=config["architecture"][x-1][2])
			print(weights[x])
			bias[x] = ls.randomize2D(rows=config["architecture"][x][2],cols=1)
			print("bias["+str(x)+"]"+"\n")
			print(bias[x])


		print("boop")

		ls.serialize(filepath=savestring + "weights.json", obj=weights)
		ls.serialize(filepath=savestring + "biases.json", obj=bias)
