"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np


def isThereCycle(matrix):
	"""
	For UNDIRECTED graphs
	It identifies if there are cycles in the provided structure (which is a matrix)
	"""
	shm = np.shape(matrix)
	if( (len(shm) != 2) or (shm[0] != shm[1]) ):
		raise NameError("Error, a (matrix) ndarray of shape (n,n) has to be provided.")

	state=[]
	#for i in range(0, len(matrix) ):
	for i in range(0, shm[0] ):
		state.append(False)
	
	#for i in range(0,len(matrix)):
	for i in range(0, shm[0] ):		
		if(not state[i]):
			if( checkCycle(i,matrix,state,-1)):
				return True

	return False


def checkCycle(index,matrix,state,ancestor):
	"""
	For UNDIRECTED graphs
	Recursive function which helps to identify if there are cycles
	"""
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



def isThereCycleDG(matrix):
	"""
	For DIRECTED graphs
	It identifies if there are cycles in the provided structure (which is a matrix)
	"""

	shm = np.shape(matrix)
	if( (len(shm) != 2) or (shm[0] != shm[1]) ):
		raise NameError("Error, a (matrix) ndarray of shape (n,n) has to be provided.")

	state=[]
	#for i in range(0, len(matrix) ):
	for i in range(0, shm[0] ):
		state.append(False)
	
	#for i in range(0,len(matrix)):
	for i in range(0, shm[0] ):		
		if(not state[i]):
			if( checkCycleDG(i,matrix,state)):
				return True

	return False


def checkCycleDG(index,matrix,state):
	"""
	For DIRECTED graphs
	Recursive function which helps to identify if there are cycles
	"""
	state[index] = True
	#print(index,state)

	for i in range(len(matrix)):
		if( (matrix[index,i] == 1) ):
			if(state[i]):
				return True
			else:
				if( checkCycleDG(i,matrix,state) ):
					return True
	state[index] = False
	return False


def isThereCycleU_D_G(matrix):
	"""
	Search for cycles/circuits formed by DIRECTED EDGES
		The graph contains directed and undirected edges

	First, the undirected edges are deleted
	second, cycles are search in the grapd with directed edges	
	"""
	isMatrix(matrix)

	shm = np.shape(matrix)
	mat = matrix.copy()

	#First undirected edges are removed
	for i in range(shm[0]):
		for j in np.where(mat[i])[0]:
			if( (j > i) and mat[j,i]):
				#remove undiredted edge
				mat[i,j] = 0
				mat[j,i] = 0

	# search cycles in the DIRECTED graPH
	return isThereCycleDG(mat)


def isUndirected(matrix):
	"""
	Return true if the graph is undirected.
	It is undirected, if matrix[x,y] == 1 and matrix[y,x]==1, for all x,y where matrix[x,y] == 1
	"""
	ismatrix(matrix)

	shm = np.shape(matrix)

	for i in range(shm[0]):
		for j in np.where(matrix[i])[0]:
			if( not matrix[j,i] ):
				return False
	return True


def isMatrix(matrix):
	"""
	check if matrix is a ndarray of shape (n,n)
	"""
	shm = np.shape(matrix)
	if( (len(shm) != 2) or (shm[0] != shm[1]) ):
		raise NameError("Error, a (matrix) ndarray of shape (n,n) has to be provided.")



if __name__=="__main__":
	"""
	mates=np.zeros((5,5)).astype(int)
	
	mates[0,1]=1
	mates[1,2]=1
	mates[2,1]=1	#ocacionaria error
	mates[2,4]=1
	mates[4,3]=1
	print("Tiene ciclo:")
	print(isThereCycle(mates))
	print("++++++++++++++++++++++")
	"""

	print("UNDIRECTED")
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
	print(isThereCycle(mates))
	
	print(mates)

	#bpp_grafo(mates)
	print("Tiene ciclo:")
	print(isThereCycle(mates))
	#print(isCycleFrom(mates,2))


	print("\nDIRECTED")
	mates=np.zeros((5,5)).astype(int)
	
	mates[0,1]=1
	#mates[1,0]=1
	print(isThereCycleDG(mates))
	mates[0,2]=1
	#mates[2,0]=1
	print(isThereCycleDG(mates))
	#mates[0,3]=1
	mates[3,0]=1
	print("nc ",isThereCycleDG(mates))
	mates[1,3]=1
	#mates[3,1]=1
	print(isThereCycleDG(mates))
	
	print(mates)

	#bpp_grafo(mates)
	print("Tiene ciclo:")
	print(isThereCycleDG(mates))
	#print(isCycleFrom(mates,2))
