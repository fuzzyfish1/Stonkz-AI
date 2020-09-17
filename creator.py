from pathlib import Path
from handyfuncs import listfuncs as lf
from handyfuncs import savefuncs as sf
import os
import sys

class AI_initer:
	
	@staticmethod
	def buildconfig(confpath = "conf.json"):
		
		origindirectory = Path(os.path.realpath(sys.argv[0])).parent
		
		confpath = origindirectory / confpath
		savedirpath = origindirectory / 'save/'
		
		config = {
			
			"save-directory": str(savedirpath)+'\\',
			
			"iterations": 0,
			"totalcost": 0,
			"architecture":
				[
					[
						"fullyconnected", "sigmoid", 14
						# perceptrons the first and second thing shouldnt matter
					],
					[
						"fullyconnected", "leaky ReLU", 14
					],
					[
						"fullyconnected", "leaky ReLU", 14
					],
					[
						"fullyconnected", "leaky ReLU", 14
					],
					[
						"fullyconnected", "leaky ReLU", 14
					],
					[
						"fullyconnected", "sigmoid", 4
					]
				]
		}
		
		sf.serialize(obj=config, filepath=str(confpath))
		
		# randomizing the neural network
		
		weights = [0 for x in range(len(config["architecture"]))]
		bias = [0 for x in range(len(config["architecture"]))]
		
		for x in range(1, len(config["architecture"])):
			
			weights[x] = lf.randomize2D(rows=config["architecture"][x][2],
			                            cols=config["architecture"][x - 1][2],
			                            min = -10,max = 10)
			bias[x] = lf.randomize2D(rows=config["architecture"][x][2], cols=1,min = -10,max = 10)
		
		sf.serialize(filepath=str(savedirpath / 'weights.json'), obj=weights)
		sf.serialize(filepath=str(savedirpath / 'biases.json'), obj=bias)