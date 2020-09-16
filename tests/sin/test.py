import numpy as np
import math
import listfuncs as ls
from Neural import AI
from Neural import AI_initer as creator
import matplotlib.pyplot as plt

#creator.buildconfig()

increment = 0.05
x1 = []
yreal =[]
ypredicted =[]
ai = AI('conf.json')

repeats = 10
for r in range(repeats):
	for x in range(-7*20,7*20):
		rad = x*increment
		print("rad: "+str(rad))
		output = math.sin(rad)
		ai.backpropagate(input = [[rad]], output = output)
		if r == 1:
			print("                                   appended real")
			x1.append(rad)
			yreal.append(output)
		
		
print("                    prediction finder                     ")
for x in range(-7*20,7*20):
	rad = x*increment
	prediction = ai.predict(data = [[rad]])[0][0]
	ypredicted.append(prediction)
'''
for z in range(-10*20,10*20):
	num = z*increment
	
	j = np.array([[num]])
	j[j <= 0] = 0.05
	j[j > 0] = 1
	
	yreal.append(np.where(num > 0, num, num * 0.05))
	ypredicted.append(j[0][0])
	x1.append(num)
'''
plt.plot(x1,yreal,'r-')
plt.plot(x1,ypredicted,'b-')
plt.show()