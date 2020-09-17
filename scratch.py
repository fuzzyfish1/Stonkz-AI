import numpy as np
import random

# randomized 1d list
def randomize1D(length=1, min = -1, max = 1 ):
	d = [random.uniform(min,max) for i in range(length)]
	return d

# randomized 2d array
def randomize2D(rows=0, cols=0, min=0, max=100,):
	c = [randomize1D(length=cols, min = min,max = max) for x in range(rows)]
	return c

z= randomize2D(rows= 3,cols=3,min = -3,max = 4)
print(z)
