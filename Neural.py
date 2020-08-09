import listfuncs as ls
import numpy as np
import json
import os
import platform


class AI:

	def squash(self,inputs):
		return 1/(1+np.exp(-inputs))

	def __init__(self,configfile):

		self.config = ls.deserialize(filepath = configfile)

		self.iterations = self.config["iterations"]
		self.savespace = self.config["save-directory"]


		try:

			self.hiddenweight1 = ls.deserialize( filepath = self.savespace+"hiddenweight1.json")
			self.hiddenbias1 = ls.deserialize( filepath = self.savespace+"hiddenbias1.json")
			self.hiddenweight2 = ls.deserialize( filepath = self.savespace+"hiddenweight2.json")
			self.hiddenbias2 = ls.deserialize( filepath = self.savespace+"hiddenbias2.json")
			print("i beleive the network init successfully")

		except:

			print("\n"+"serialization failure"+"\n"+"randomizing")
			self.hiddenweight1 = np.array(ls.randomize2D(rows=7,cols=3))
			self.hiddenbias1= np.array(ls.randomize2D(rows=7,cols=1))
			self.hiddenweight2=np.array(ls.randomize2D(rows=3,cols=7))
			self.hiddenbias2=np.array(ls.randomize2D(rows=3,cols=1))

		finally:
			print("AI initialized sucessfully")


	def save(self):

		ls.serialize( filepath = self.savespace + "hiddenweight1.json" , obj = self.hiddenweight1.tolist())
		ls.serialize( filepath = self.savespace + "hiddenbias1.json" , obj = self.hiddenbias1.tolist())
		ls.serialize( filepath = self.savespace + "hiddenweight2.json" , obj = self.hiddenweight2.tolist())
		ls.serialize( filepath = self.savespace + "hiddenbias2.json" , obj = self.hiddenbias2.tolist())

	def printweights():

		print("\n\n"+"hiddenweight1"+"\n\n")
		print(self.hiddenweight1)
		print("\n\n"+"hiddenbias1"+"\n\n")
		print(self.hiddenbias1)


	def predict(self, data=[[4],[-300],[100]]):

		print("predicting")

		perceptrons = self.squash(np.array(data))    #3rows 1 cols

		self.activations = [perceptrons]
		self.z = [0]
		self.weights = [0,self.hiddenweight1,self.hiddenweight2]
		self.bias = [0,self.hiddenbias1,self.hiddenbias2]

# x starts from 1
# x = 0 is perceptrons
# x is the activation being calculated

		for x in range(1,len(self.activations)):

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

		print("\n\n"+"**learning**"+"\n\n")
		answer = np.array(self.predict(input))
		correct = np.array(output)

		cost = (answer - correct)**2

		self.C = []

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

		self.weights = self.newweights
		self.bias = self.newbias

		self.save()

"""
		dsig2 = dsigmoid(self.z2)

		delCdelw2 = np.dot(np.multiply(C,dsig2),np.transpose(self.activations1))

		delCdelb2= np.multiply(C,dsig2)

		delCdela1 = np.dot(np.transpose(np.multiply(self.hiddenweight2,dsig2)),C)

		dsig1= dsigmoid(self.z1)

		delCdelw1 = np.dot(np.multiply(delCdela1,dsig1), np.transpose(ins))

		delCdelb1 = np.multiply(delCdela1,dsig1)

#<<<<<<<<<-----------------------------------------------------changing the weights----------------------------------------------->>>>>>

		self.hiddenbias2=  np.add(self.hiddenbias2,-delCdelb2)

		self.hiddenweight2 =np.add(self.hiddenweight2,-delCdelw2)

		self.hiddenbias1 = np.add(self.hiddenbias1,-delCdelb1)

		self.hiddenweight1 = np.add(self.hiddenweight1,-delCdelw1)

		print("\n\n"+"<<<-------------------------------cost-------------------------------------->>>"+"\n\n")

		print(cost)

		self.save()
"""

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






"""
#		print("this prolly needs admin priveleges")

		plat = platform.system()

		if("Linux" in plat):

			print("I have detected a linux based computer")

			saved = False

			while(saved is False):

				try:
					savespot= input("enter directory to create save directory")
					os.system('sudo mkdir '+savespot+'/save')
					os.system('sudo chmod 777 '+savespot+'/save')
					saved = True
				except:
					print("i dont think it worked try again")

			config = {

				"save-directory" : savespot+'/save',

				"iterations" : 0

			}

			ls.serialize(obj =config,filepath = "Stonkz-AI-config.json")


		elif("Windows" in plat):
			print("I have detected a Windoes based computer")

		elif("Darwin" in plat):
			print("I have detected a Mac")
			print("Macs are overpriced and shitty seriously get rid of it")

"""









# step 1 is to successfully propogate 2 layers                       ---------completed----------
# step 2 is to save/freeze network				     ---------completed----------
# step 3 is to re init the network from files			     ---------completed----------
# step 4 is to backpropogate 2 layers				     ---------completed----------
# step 5 is
#	a chungus cleaning
#	config file
# step 6 is
#	heavy remathing
#	big renaming of everything
# step 7 is external storage support				     ---------completed----------
# step 8 is to turn the propagations into a step forward+ step back
# step 9 is to propogate through a list of layers
# step 10 is to backpropogate through a list of layers
# step 11 is to create an instantiation method
