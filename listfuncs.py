import random as random
import json

# randomized 1d list
def randomize1D(length=1,coefficient = 1):

    k = -1
    if(random.random()>.5):
        k = 1
    else:
        k = -1
    d = [random.random()*coefficient*k for i in range(length)]
    return d

# randomized 2d array
def randomize2D(rows=0, cols=0, min=0, max=100, coefficient=1):
    c = [randomize1D(length=cols,coefficient=coefficient) for x in range(rows)]
    return c

# filled 2d list
def fill1D (num=0,length =1):
    c= [num for x in range(length)]
    return c

# filled 2d array
def fill2D(num= 0,rows=1,cols=1):
    c = [fill1D(num=num,length =cols) for x in range(rows)]
    return c

# saves a obj as a json
def serialize(filepath,obj):
	with open(filepath, "w+") as file:
		json.dump(obj, file, indent=4, separators=(',', ':'))

# creates a json obj from a json file
def deserialize(filepath):
	with open(filepath, "r") as file:
		return json.load(file)

# converts 2d matrices into vectors
def vectorize(obj):
    # takes a list and vectors it

    c =[]

    for o in obj:
        if type(o) is type(1):
            c.append([o])
        else:
            for l in o:
                c.append([l])
    return c

# return true if its a vector false if anything else
def isvector(obj):
    for o in obj:
        try:
            c = len(o)
        except:
            c = 5
            return False
            break
        finally:
            if c != 1:
                return False
                break
    return True

# backsup a file
def backup(filepath_of_backup, filepath):
    object = deserialize(filepath=filepath_of_backup)
    serialize(filepath = filepath,obj = object)


