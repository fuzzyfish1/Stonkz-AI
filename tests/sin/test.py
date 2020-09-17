import math
from Neural import AI
from creator import AI_initer as creator
import matplotlib.pyplot as plt

spot = 'C:\\Users\\uwish\\PycharmProjects\\Stonkz-AI\\tests\\sin\\test2.json'
creator.buildconfig(spot)

increment = 0.05
x1 = []
yreal =[]
ypredicted =[]
ai = AI(spot)

repeats = 10
for r in range(repeats):
	
	for x in range(-7*20,7*20):
		rad = x*increment
		print("rad: "+str(rad))
		output = math.sin(rad)
		ys.append(ai.predict(data = [[rad]])[0][0])
		ai.backpropagate(output = output)
		if r == 1:
			print("                                   appended real")
			x1.append(rad)
			yreal.append(output)
		
		
print("                    prediction finder                     ")
for x in range(-7*20,7*20):
	rad = x*increment
	prediction = ai.predict(data = [[rad]])[0][0]
	ypredicted.append(prediction)

plt.plot(x1,yreal,'r-')
plt.plot(x1,ypredicted,'b-')
plt.show()