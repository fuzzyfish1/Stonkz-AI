#import numpy as np
from Neural import AI
from Neural import AI_initer as creator
import listfuncs as ls
import random as rand


print("test.py initiated")

#creator.buildconfig()

ai = AI('Stonkz-AI-config.json')

print(ai.predict())

for x in range(100):

	ai.backpropagate([[1],[0],[0]], [[0],[0],[0]])
