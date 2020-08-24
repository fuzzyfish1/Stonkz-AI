import json
import random
import numpy as np

# i beleive 4 list these methods work
def randomize1D(length=1):
    d = [random.random() for i in range(length)]
    return d

def randomize2D(rows=0, cols=0, min=0, max=100, coefficient=.01):
    c = [randomize1D(length=cols) for x in range(rows)]
    return c

def fill1D (num=0,length =1):
    c= [num for x in range(length)]
    return c

def fill2D(num= 0,rows=1,cols=1):
    c = [fill1D(num=num,length =cols) for x in range(rows)]
    return c

class AI:

    # tested works
    def serialize(self, file, obj):
        with open(file, "w+") as somefile:
            json.dump(obj, somefile, indent=4, separators=(',', ':'))

    # tested works
    def deserialize(self, file):
        with open(file, "r") as some1file:
            return json.load(some1file)

    def __init__(self):
        pass

        '''
        self.alllayer.append(self.randomize1D(length=30*30))
        self.alllayer = self.alllayer+(self.randomize2D(row = 5,cols =5))
        self.alllayer.append(self.randomize1D(length =4))
        self.serialize()
        self.deserialize()
        '''

    def predict(self, ins=[]):
        perceptrons = np.array(self.deserialize("perceptrons.json"))
        if len(ins) == len(perceptrons):

            print("\n yo we predicting")

            '''
            weights1to2 = np.array(randomize2D(rows=900, cols=5))
            bias2 = np.array(randomize2D(rows=1, cols=5))
            weights2to3 = np.array(randomize2D(rows=5,cols=7))
            bias3 = np.array(randomize2D(rows =1,cols = 7))
            weights3to4 = np.array(randomize2D(rows = 7,cols=4))
            '''

            weights1to2 = np.array(self.deserialize("weights1to2.json"))
            bias2 = np.array(self.deserialize("bias2.json"))
            weights2to3 = np.array(self.deserialize("weights2to3.json"))
            bias3 = np.array(self.deserialize("bias3.json"))
            weights3to4 = np.array(self.deserialize("weights3to4.json"))

            perceptrons = np.array(ins)
            perceptrons = 1 / (1 + np.exp(-perceptrons))

            activations2 = np.dot(perceptrons, weights1to2)
            activations2 = np.add(activations2, bias2)
            activations2 = 1 / (1 + np.exp(-activations2))

            activations3 = np.dot(activations2, weights2to3)
            activations3 = np.add(activations3, bias3)
            activations3 = 1 / (1 + np.exp(-activations3))

            output = np.dot(activations3, weights3to4)
            output = 1 / (1 + np.exp(-output))

            return output

        else:
            print("input sizing failure")
            return [0, 0, 0, 0]

        '''
        self.serialize(obj = perceptrons.tolist(),file =  'perceptrons.json')
        self.serialize(obj = weights1to2.tolist(),file =  'weights1to2.json')
        self.serialize(obj = bias2.tolist(),file =  'bias2.json')
        self.serialize(obj = weights2to3.tolist(),file =  'weights2to3.json')
        self.serialize(obj = bias3.tolist(),file =  'bias3.json')
        self.serialize(obj = weights3to4.tolist(),file =  'weights3to4.json')

        self.deserialize(file =  'perceptrons.json')



        pass
        '''

    def learn(self,cost, ins=[]):

        learningjunk = self.deserialize(file="neural.json")

        # catches most errors as to if any of the types are wrong or unreasonable except if there is nothing in the file
        # or if iterations are less than 0
        if ((isinstance(learningjunk,dict) is False)\
                or (isinstance(learningjunk.get("totalcost"),(float,int)) is False) \
                or (isinstance(learningjunk.get("iterations"),(int)) is False) \
                or (isinstance(learningjunk.get("avgcost"),(float,int)) is  False)):
            learningjunk = {"totalcost": 0, "iterations": 0, "avgcost": 0}
            self.serialize(obj=learningjunk,file="neural.json")

        iterations = learningjunk["iterations"]
        totalcost = learningjunk["totalcost"]

        totalcost += cost
        iterations += 1
        avgcost = totalcost / iterations

        perceptrons = np.array(self.deserialize("perceptrons.json"))
        if len(ins) == len(perceptrons):

            print("\n predicting for learning")

            '''
            weights1to2 = np.array(randomize2D(rows=900, cols=5))
            bias2 = np.array(randomize2D(rows=1, cols=5))
            weights2to3 = np.array(randomize2D(rows=5,cols=7))
            bias3 = np.array(randomize2D(rows =1,cols = 7))
            weights3to4 = np.array(randomize2D(rows = 7,cols=4))
            '''

            weights1to2 = np.array(self.deserialize("weights1to2.json"))
            bias2 = np.array(self.deserialize("bias2.json"))
            weights2to3 = np.array(self.deserialize("weights2to3.json"))
            bias3 = np.array(self.deserialize("bias3.json"))
            weights3to4 = np.array(self.deserialize("weights3to4.json"))

            perceptrons = np.array(ins)
            perceptrons = 1 / (1 + np.exp(-perceptrons))

            activations2 = np.dot(perceptrons, weights1to2)
            activations2 = np.add(activations2, bias2)
            activations2 = 1 / (1 + np.exp(-activations2))

            z3 = np.dot(activations2, weights2to3)
            z3 = np.add(z3, bias3)
            activations3 = 1 / (1 + np.exp(-z3))

            z4 = np.dot(activations3, weights3to4)
            output = 1 / (1 + np.exp(-z4))

            print("backpropagating")

            if cost <= 0:
                #to obtain the desired values
                if ((output[0][0] > output[0][1]) and (output[0][0] > output[0][2]) and(output[0][0] > output[0][3])):
                    real=np.array([[1,0,0,0]])
                elif ((output[0][1] > output[0][0]) and (output[0][1] > output[0][2]) and(output[0][1] > output[0][3])):
                    real=np.array([[0,1,0,0]])
                elif ((output[0][2] > output[0][0]) and (output[0][2] > output[0][1]) and(output[0][2] > output[0][3])):
                    real=np.array([[0,0,1,0]])
                elif ((output[0][3] > output[0][0]) and (output[0][3] > output[0][1]) and(output[0][3] > output[0][2])):
                    real=np.array([[0,0,0,1]])
                else:
                    pass

                # compute backpropagation

                costs = np.multiply((output -real),(output -real))

                differencecost = np.array(output-real)

                dsigz4= 2*(np.exp(-z4)/np.multiply((1+np.exp(-z4)),(1+np.exp(-z4))))# derivative of squish z4

                #dsigz4[0][0]

                print("dsigz4")
                print(dsigz4)
                print("\n\n\n")
                num1 = np.multiply(differencecost,dsigz4)

                inputmatrix = np.array(fill1D(activations3[0].tolist(),length=len(weights3to4[0])))

                print(weights3to4)
                print("\n\n\n")
                print(inputmatrix)

                gradientw3to4 = np.dot(2*num1,activations3)

                weights3to4 = weights3to4 -gradientw3to4

            learningjunk = {"totalcost": totalcost, "iterations": iterations, "avgcost": avgcost}

            self.serialize(obj=learningjunk, file="neural.json")

            #ye
