#import numpy as np
from Neural import AI
from Neural import AI_initer as creator
import numpy as np
import listfuncs as ls
import random as rand
import matplotlib.pyplot as plt

print("test.py initiated")
'''
x= []
y =[]
x2 =[]
y2 = []
rows = 15
columns = 15
lastdiscountfactor = 0
discountfactor =0
for steps in range(1,14**2):
	x.append(steps)
	x2.append(steps)
	discountfactor = 1-(steps/(14**2))
	y.append(discountfactor)
	y2.append(discountfactor-lastdiscountfactor)
	print((discountfactor - lastdiscountfactor)*15)
	lastdiscountfactor = discountfactor
	
plt.plot(x,y)
plt.plot(x2,y2,'r-')


plt.show()
'''
creator.buildconfig()