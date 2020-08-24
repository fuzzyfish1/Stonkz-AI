#import numpy as np
from Neural import AI
from Neural import AI_initer as creator
import numpy as np
import listfuncs as ls
import random as rand


print("test.py initiated")

#creator.buildconfig()

creator.buildconfig()
'''
z =ls.randomize2D(rows= 3,cols=3)

print("\n"+"z: "+"\n\n")
print(z)

def sigmoid(inputs):
    return np.reciprocal(1+np.exp(np.multiply(-1,inputs)))

y =sigmoid(z)
#print(y)

g = ls.vectorize(z)
print("\n"+"behold vectorized z :"+"\n\n")
print(g)
print("g is: ")
f = ls.isvector(g)
print(f)
d = ls.isvector(z)
print("z is: ")
print(d)
'''