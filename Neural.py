import listfuncs as ls
import numpy as np
import json
import os
import platform


class AI:

	def squash(self,inputs):
		return 1/(1+np.exp(-inputs))

	def __init__(self,configfile):

		self.configpath = configfile
		self.config = ls.deserialize(filepath = configfile)

		self.iterations = self.config["iterations"]
		self.savespace = self.config["save-directory"]

		print("savespace is "+ self.savespace)

#		try:

		self.halfserializedweights = ls.deserialize(filepath = self.savespace+ "weights.json")

		self.weights = self.halfserializedweights

		for w in range(len(self.halfserializedweights)):
			self.weights[w] = np.array(self.halfserializedweights[w])


		self.halfserializedbiases = ls.deserialize(filepath = self.savespace + "biases.json")

		self.bias= self.halfserializedbiases

		for b in range(len(self.halfserializedbiases)):

			self.bias[b] = np.array(self.halfserializedbiases[b])

		print("i beleive the network init successfully")

#		except:

#			print("\n"+"serialization failure"+"\n"+"FFFFuuuuuuuckkkk")

#		finally:
#			print("AI initialized sucessfully")


	def save(self):

#		ls.serialize( filepath = self.savespace + "hiddenweight1.json" , obj = self.hiddenweight1.tolist())
#		ls.serialize( filepath = self.savespace + "hiddenbias1.json" , obj = self.hiddenbias1.tolist())
#		ls.serialize( filepath = self.savespace + "hiddenweight2.json" , obj = self.hiddenweight2.tolist())
#		ls.serialize( filepath = self.savespace + "hiddenbias2.json" , obj = self.hiddenbias2.tolist())


# turning everything into a list in a list in a list to save to 1 file

		self.saveweights = self.weights

		for x in range(len(self.weights)):
			print(x)
			self.saveweights[x] = self.weights[x].tolist()

		self.savebiases = self.bias

		for b in range(len(self.bias)):
			self.savebiases[b] = self.bias[b].tolist()

		ls.serialize(filepath = self.savespace +"weights.json", obj = self.saveweights)
		ls.serialize(filepath = self.savespace +"biases.json", obj = self.savebiases)

		config = {
			"save-directory" : self.savespace,
			"iterations" : self.iterations
		}

		ls.serialize( filepath = self.configpath, obj = config)

	def printweights(self):

		for w in self.weights:
			print(w)

		for b in self.bias:
			print(b)

	def predict(self, data=[[4],[-300],[100]]):

		self.printweights()
		print("predicting")

		perceptrons = self.squash(np.array(data))    #3rows 1 cols

		self.activations = [perceptrons]
		self.z = [np.array([[0]])]

		# x starts from 1
		# x = 0 is perceptrons
		# x is the activation being calculated

		for x in range(1,len(self.activations)):
			print("thing")
			self.z[x] = np.add(np.dot(weights[x],self.activations[x-1]),self.bias[x])
			self.activations[x] = self.squash(self.z[x])
			print("calculating:")
			print(x)



		answer = self.activations[-1]




		return answer




		'''
		x=[[1,0],[0,1]]
		a = [[4,1],[2,2]]
		2d matrices
		each inside list is its own row starting from the top ex:

		x looks like 1 0	a looks like 4 1
			     0 1		     2 2

		np.dot(x,a)= a

		'''

		# cols of the left must match the rows of the left

		#self.hiddenweight2=np.array(ls.randomize2D(rows=3,cols=7))	   		#3 rows 7cols
		#self.hiddenbias2=np.array(ls.randomize2D(rows=3,cols=1))	   		#3 rows 1 cols
		"""
		self.z1 = np.add(np.dot(self.hiddenweight1,self.perceptrons),self.hiddenbias1)

		self.activations1 = self.squash(self.z1)

		self.z2 = np.add(np.dot(self.hiddenweight2,self.activations1),self.hiddenbias2)

		self.activations2 = self.squash(self.z2)

		answer = self.activations2.tolist()
		"""





	def backpropagate(self,input = [[1],[0.2],[0.8]], output=[[.3],[.2],[.1]]):


		ins = np.array(input)

		def dsigmoid(z):
			return np.exp(-z)/((1+np.exp(-z))**2)


		# inputs can only be a column of 3

		print("\n\n"+"**learning**"+"\n"+"iteration:")
		print(self.iterations)

		answer = np.array(self.predict(input))
		correct = np.array(output)

		cost = (answer - correct)**2

		self.C = [ np.array([[0],[0]]) for x in range(len(self.activations)+1)]

		self.C[len(self.activations)] = 2*(answer-correct)

#		C = del C/del cost

#		behold my fucking math

		self.dsig = [0]
		self.newweights = [0]
		self.newbias = [0]

		for x in range(0,len(self.activations)-1):

			l = len(self.activations) - x
			self.dsig[l]= dsigmoid(self.z[l])
			self.newweights[l] = np.dot(np.multiply(self.C[l],self.dsig[l]),np.transpose(self.activations[l]))
			self.newbias[l] = np.multiply(self.C[l],self.dsig[l])
			self.C[l-1] = np.dot(np.transpose(np.multiply(self.weights[l],self.dsig[l])),self.C[l])

		self.iterations += 1
		self.weights = self.newweights
		self.bias = self.newbias

		self.save()




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
			"iterations" : 0
		}

		ls.serialize(obj =config,filepath = savespot)













# we need to save the weights and biases in one file
# we need to try to add ReLU neurons
# we need to add convolution layers
# we need to add pooling layers
# we need to begin finding API for stock data calls
