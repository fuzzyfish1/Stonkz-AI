import random as random
import json

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

def serialize(filepath,obj):
	with open(filepath, "w+") as file:
		json.dump(obj, file, indent=4, separators=(',', ':'))

def deserialize(filepath):
	with open(filepath, "r") as file:
		return json.load(file)
