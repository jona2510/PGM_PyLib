"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

#libraries
import numpy as np
import PGM_PyLib.utils as utils	
import PGM_PyLib.graphCycle as gc	


class CLP_MI:
	"""
	This method is based in the idea of the Chow and Liu procedure
	which uses "Mutual Information" in order to build the tree
	"""

	def __init__(self, root=0, heuristic=False, smooth=0.1):

		self.root = root
		self.heuristic = heuristic
		self.smooth = smooth

		self.structure = [] 	# a matrix which contains the structure
		self.labels = []		# labels of the matrix ## quiza no es necesario
		return

	def createStructure(self, data):			
		"""
		 data is the matrix of data, the columns are the attributes
		 root: is the index of the attribute which will be selected as the root of the tree (can be an int random value)
		 I: can be utils.MI or utils.CMI
		 Z: is used if CMI was selected
		 heuristic: True to use the heuristic to select the root node (The parameter "root is ignored")
		"""

		n = len(data[0])
		self.structure = np.zeros(( n , n )).astype(int)


		"""
		Below the MI/CMI are estimaed between each pair (x,y) where x!=y

		info[0] has index
		info[1] has index
		info[2] has the value computed by "I"
		"""
		info = np.zeros(( 3, int( ( n*(n - 1) ) / 2 )  ))
		
		c=0
		for i in range(n):
			for j in range(i+1,n):
				info[0,c] = i
				info[1,c] = j
				
				#if(I == "MI"):
				info[2,c] = utils.MI( data[:,i], data[:,j], self.smooth )
				#elif(I == "CMI"):
				#	info[2,c] = utils.CMI( data[:,i], data[:,j], Z, 0.01 )
				c += 1

		# it orders the scores asc
		info=info[ :, info[2,:].argsort() ]

		# edges will be added to the structure avoiding cycles
		state = np.zeros(n)	# state of the nodes, 1 = visited, 0 = no visited
		c = 0
		while( (np.sum(state) != n) and (c < len(info[0])) ):
			self.structure[ int(info[0,c]) , int(info[1,c]) ] = 1
			self.structure[ int(info[1,c]) , int(info[0,c]) ] = 1

			if( gc.isThereCycle(self.structure) ):	
				#if the insertion of the edge generates a cycle, then the edge is deleted
				self.structure[ int(info[0,c]) , int(info[1,c]) ] = 0
				self.structure[ int(info[1,c]) , int(info[0,c]) ] = 0
			else:
				# if the edge does not generate a cycle, then the nodes are "visited"
				state[ int(info[0,c]) ] = 1
				state[ int(info[1,c]) ] = 1

			c += 1

		#print("value c: "+str(c))

		self.giveDirections(self.heuristic)

		return self.structure.copy()


	"""
	This function assigns directions to the undirected structure (a matrix)
	The first attribute will be selected as the root, however a random value can be used
		which will be procesed in order to fit with the size of the matrix
	On the other hand, a heuristic procedure for selecting the root is proposed
		which selects as root the node with more conections to other nodes
	"""
	def giveDirections(self, heuristic=False ):

		if(heuristic):
			# it selects as root the node with more conecctions to other nodes
			# if there are more than one with the same maximum quantity, it selects the first of them
			bonds = sum(self.structure)		#bonds - links - connections 	# it is uselful because the structure is undirected
			#bonds=[sum( f[i] ) for i in range( len(f) )]	# descendents 		

			rootl = np.where( bonds == max(bonds) )[0][0]
			self.root = rootl
		else:
			rootl = int( round( self.root ) % len(self.structure) )	

		state = np.zeros(len(self.structure))	

		self.giveDirectionsRec(rootl,state)	

	def giveDirectionsRec(self, index, state):
		state[index] = 1

		for i in range(len(self.structure)):
			if( (self.structure[index,i] == 1) and (state[i] == 0) ):
				self.structure[i, index] = 0	# it deletes that direction

				self.giveDirectionsRec(i,state) 
				

class CLP_CMI(CLP_MI):
	"""
	This method is based in the idea of the Chow and Liu procedure
	which uses "Conditional Mutual Information" in order to build the tree
	"""

	def createStructure(self, data, Z):
		"""
		 data is the matrix of data, the columns are the attributes
		 root: is the index of the attribute which will be selected as the root of the tree (can be an int random value)
		 I: can be utils.MI or utils.CMI
		 Z: is used if CMI was selected
		 heuristic: True to use the heuristic to select the root node (The parameter "root is ignored")
		"""

		if( len(Z) != len(data) ):
			raise NameError("the number of elements in data is different from Z")			

		n = len(data[0])
		self.structure = np.zeros(( n , n )).astype(int)


		"""
		Below the MI/CMI are estimaed between each pair (x,y) where x!=y

		info[0] has index
		info[1] has index
		info[2] has the value computed by "I"
		"""
		info = np.zeros(( 3, int( ( n*(n - 1) ) / 2 )  ))
		
		c=0
		for i in range(n):
			for j in range(i+1,n):
				info[0,c] = i
				info[1,c] = j
				
				#if(I == "MI"):
				#	info[2,c] = utils.MI( data[:,i], data[:,j], 0.01 )
				#elif(I == "CMI"):
				info[2,c] = utils.CMI( data[:,i], data[:,j], Z, self.smooth )
				c += 1

		# it orders the scores asc
		info=info[ :, info[2,:].argsort() ]

		# edges will be added to the structure avoiding cycles
		state = np.zeros(n)	# state of the nodes, 1 = visited, 0 = no visited
		c = 0
		while( (np.sum(state) != n) and (c < len(info[0])) ):
			self.structure[ int(info[0,c]) , int(info[1,c]) ] = 1
			self.structure[ int(info[1,c]) , int(info[0,c]) ] = 1

			if( gc.isThereCycle(self.structure) ):	
				#if the insertion of the edge generates a cycle, then the edge is deleted
				self.structure[ int(info[0,c]) , int(info[1,c]) ] = 0
				self.structure[ int(info[1,c]) , int(info[0,c]) ] = 0
			else:
				# if the edge does not generate a cycle, then the nodes are "visited"
				state[ int(info[0,c]) ] = 1
				state[ int(info[1,c]) ] = 1

			c += 1

		#print("value c: "+str(c))

		self.giveDirections(self.heuristic)

		return self.structure.copy()

			
