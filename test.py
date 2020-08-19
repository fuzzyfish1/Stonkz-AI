#import numpy as np
from Neural import AI
from Neural import AI_initer as creator
import listfuncs as ls
import random as rand


print("test.py initiated")

creator.buildconfig()

ai = AI('conf.json')

print()
print(ai.predict([[1],[1],[1],[1],[1],[1],[1]]))
for z in range(100):
    ai.backpropagate(input=[[1],[1],[1],[1],[1],[1],[1]],output= [1])

'''
for x in range(100):

	ai.backpropagate([[1],[0],[0]], [[0],[0],[0]])
'''