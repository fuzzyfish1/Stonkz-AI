#import numpy as np
from Neural import AI
from Neural import AI_initer as creator
import listfuncs as ls
import random as rand


print("test.py initiated")

#creator.buildconfig()

ai = AI('conf.json')

ai.printweights()

print(ai.predict([[1],[1]]))

for z in range(1):
    ai.backpropagate(input=[[.45],[.43]],output= [[0],[.5],[.3]])

'''
for x in range(100):

	ai.backpropagate([[1],[0],[0]], [[0],[0],[0]])
'''