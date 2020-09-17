import json
from handyfuncs import listfuncs as ls
import numpy as np

# saves a obj as a json
def serialize(filepath, obj):
	with open(filepath, "w+") as file:
		json.dump(obj, file, indent=4, separators=(',', ':'))

# saves a list of numpy arrays as a list
def savelistofarrays(filepath, obj):
	saveobj = obj
	
	for x in range(len(obj)):
		
		if (type(obj[x]) is int):
			saveobj[x] = obj[x]
		else:
			saveobj[x] = obj[x].tolist()
	
	serialize(filepath=filepath, obj=saveobj)

# loads everything into a list of arrays
def loadlistofarrays(filepath):
	obj1 = deserialize(filepath=filepath)
	obj2 = obj1
	
	for w in range(len(obj1)):
		obj2[w] = np.array(obj1[w])
		
	return obj2

# creates a json obj from a json file
def deserialize(filepath):
	with open(filepath, "r") as file:
		return json.load(file)

# backsup a file
def backup(filepath_of_backup, filepath):
	object = deserialize(filepath=filepath_of_backup)
	serialize(filepath=filepath, obj=object)
