import random as random
import json
import numpy as np

# randomized 1d list
def randomize1D(length=1, min = -1, max = 1 ):
	d = [random.uniform(min,max) for i in range(length)]
	return d

# randomized 2d array
def randomize2D(rows=0, cols=0, min=0, max=100,):
	c = [randomize1D(length=cols, min = min,max = max) for x in range(rows)]
	return c

# filled 2d list
def fill1D(num=0, length=1):
	c = [num for x in range(length)]
	return c

# filled 2d array
def fill2D(num=0, rows=1, cols=1):
	c = [fill1D(num=num, length=cols) for x in range(rows)]
	return c

# converts 2d matrices into vectors
def vectorize(obj):
	# takes a list and vectors it
	
	c = []
	
	for o in obj:
		if type(o) is type(1):
			c.append([o])
		elif type(o) is type("junk"):
			c.append([o])
		else:
			for l in o:
				c.append([l])
	return c

# return true if its a vector false if anything else
def isvector(obj):
	isvector = True
	
	if ((type(obj) is not list) and (type(obj) is not type(np.array([[1, 0], [0, 3]])))):
		isvector = False
	else:
		try:
			for o in obj:
				if (len(o) == 1):
					pass
				else:
					isvector = False
					break
		except:
			isvector = False

	return isvector

# given a upper and lower cap can range a numpy array to fit between
def capfunc(x, uppercap,lowercap):
	j = x
	j[j < lowercap] = lowercap
	j[j > uppercap] = uppercap
	return j
