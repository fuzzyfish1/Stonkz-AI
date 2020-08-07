import listfuncs as ls
import numpy as np
import json

class AI:

#	self.savespace = "/mnt/mydisk/save/"

#     path to disk for speedier writing/mnt/mydisk


	def squash(self,inputs):
		return 1/(1+np.exp(-inputs))

	def __init__(self):

		self.savespace = "/mnt/mydisk/save/"
		self.save = False
		self.killbool = False

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



	def build(self):
		pass

	def save(self):

		ls.serialize( filepath = self.savespace + "hiddenweight1.json" , obj = self.hiddenweight1.tolist())
		ls.serialize( filepath = self.savespace + "hiddenbias1.json" , obj = self.hiddenbias1.tolist())
		ls.serialize( filepath = self.savespace + "hiddenweight2.json" , obj = self.hiddenweight2.tolist())
		ls.serialize( filepath = self.savespace + "hiddenbias2.json" , obj = self.hiddenbias2.tolist())

	def printweights():

#		print("\n\n"+"perceptrons"+"\n\n")
#		print(perceptrons)
		print("\n\n"+"hiddenweight1"+"\n\n")
		print(self.hiddenweight1)
		print("\n\n"+"hiddenbias1"+"\n\n")
		print(self.hiddenbias1)


	def predict(self, data=[[4],[-300],[100]]):

		print("predicting")

		self.perceptrons = self.squash(np.array(data))    #3rows 1 cols

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


#		print("\n\n"+"perceptrons"+"\n\n")
#		print(perceptrons)

		#self.hiddenweight2=np.array(ls.randomize2D(rows=3,cols=7))	   		#3 rows 7cols
		#self.hiddenbias2=np.array(ls.randomize2D(rows=3,cols=1))	   		#3 rows 1 cols

#		print("\n\n"+"activations1a"+"\n\n")

		self.activations1a = np.dot(self.hiddenweight1,self.perceptrons)

#		print(activations1a)
#		print("\n\n"+"activations1b"+"\n\n")

		self.activations1b = np.add(self.activations1a,self.hiddenbias1)

#		print(activations1b)
#		print("\n\n"+"activations1c"+"\n\n")

		self.activations1c = self.squash(self.activations1b)

#		print(activations1c)

#		print("\n\n\n\n"+"-------------------propogation layer 2-----------------------"+"\n\n\n\n")


#		print("hiddenweight2"+"\n\n")
#		print(self.hiddenweight2)

#		print("\n\n"+"hiddenbias2"+"\n\n")
#		print(self.hiddenbias2)

#		print("\n\n"+"activations2a"+"\n\n")

		self.activations2a = np.dot(self.hiddenweight2,self.activations1c)
#		print(activations2a)

#		print("\n\n"+"activations2b"+"\n\n")

		self.activations2b = np.add(self.activations2a,self.hiddenbias2)
#		print(activations2b)

#		print("\n\n"+"activations2c"+"\n\n")

		self.activations2c = self.squash(self.activations2b)
#		print(activations2c)

		answer = self.activations2c.tolist()


		return answer

	def backpropagate(self,input = [[1],[0.2],[0.8]], output=[[.3],[.2],[.1]]):

		def dsigmoid(z):
			return np.exp(-z)/((1+np.exp(-z))**2)


		# inputs can only be a column of 3

		ins = np.array(input)

		print("\n\n"+"**learning**"+"\n\n")
		answer = np.array(self.predict(input))
		correct = np.array(output)

		cost = (answer- correct)**2
#		print("the cost is:")
#		print(cost)

		C = (answer-correct)

		#gradient calculation for second layer

#		print("\n\n"+"delCdelw2"+"\n\n")

		dsig2 = dsigmoid(self.activations2b)
		delCdelw2 = np.dot(np.multiply(C,dsig2),np.transpose(self.activations1c))

#		print("\n\n"+"delCdelb2"+"\n\n")

		delCdelb2= 2* np.multiply(C,dsig2)

#		print(delCdelb2)

#		print("\n\n"+"delCdela1"+"\n\n")

		delCdela1 = 2* np.dot(np.transpose(np.multiply(self.hiddenweight2,dsig2)),C)

#		print(delCdela1)

#		print("\n\n"+"delCdelw1"+"\n\n")

		dsig1= dsigmoid(self.activations1b)

		delCdelw1 = np.dot(np.multiply(delCdela1,dsig1), np.transpose(ins))

#		print(delCdelw1)

#		print("\n\n"+"delCdelb1"+"\n\n")

		delCdelb1 = np.multiply(delCdela1,dsig1)

#		print(delCdelb1)

#<<<<<<<<<-----------------------------------------------------changing the weights----------------------------------------------->>>>>>.

#		print("\n\n"+"showing the new weights now"+"\n\n")

#		print("\n\n"+"new bias2"+"\n\n")

		self.hiddenbias2=  np.add(self.hiddenbias2,-delCdelb2)

#		print("\n\n"+"new weights2"+"\n\n")

		self.hiddenweight2 =np.add(self.hiddenweight2,-delCdelw2)

#		print("\n\n"+"new bias1"+"\n\n")

		self.hiddenbias1 = np.add(self.hiddenbias1,-delCdelb1)

		self.hiddenweight1 = np.add(self.hiddenweight1,-delCdelw1)

		print("\n\n"+"<<<-------------------------------cost-------------------------------------->>>"+"\n\n")

		print(cost)


		self.save = True

		self.save()














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
