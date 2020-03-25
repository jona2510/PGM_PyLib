"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np

"""
It identifies if there are cycles in the provided structure (which is a matrix)
"""
def isThereCycle(matrix):
	state=[]
	for i in range(0,len(matrix)):
		state.append(False)
	
	for i in range(0,len(matrix)):	
		if(not state[i]):
			if( checkCycle(i,matrix,state,-1)):
				return True

	return False

"""
Recursive function which helps to identify if there are cycles
"""
def checkCycle(index,matrix,state,ancestor):
	state[index] = True
	#print(index)	#it prints BPP

	for i in range(len(matrix)):
		if( (matrix[index,i] == 1) and (i != ancestor) ):
			if(state[i]):
				return True
			else:
				if( checkCycle(i,matrix,state,index) ):
					return True
	return False


if __name__=="__main__":
	"""
	mates=np.zeros((5,5)).astype(int)
	
	mates[0,1]=1
	mates[1,2]=1
	mates[2,1]=1	#ocacionaria error
	mates[2,4]=1
	mates[4,3]=1
	"""
	mates=np.zeros((5,5)).astype(int)
	
	mates[0,1]=1
	mates[1,0]=1
	print(isThereCycle(mates))
	mates[0,2]=1
	mates[2,0]=1
	print(isThereCycle(mates))
	mates[0,3]=1
	mates[3,0]=1
	print(isThereCycle(mates))
	mates[1,3]=1
	mates[3,1]=1
	
	print(mates)

	#bpp_grafo(mates)
	print("Tiene ciclo:")
	print(isThereCycle(mates))
	#print(isCycleFrom(mates,2))
