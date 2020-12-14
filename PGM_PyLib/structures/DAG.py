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

from itertools import combinations
#from PGM_PyLib.utils import chi2_test
from PGM_PyLib.stat_tests.ci_test import chi2_ci_test
import PGM_PyLib.graphCycle as gc	


class PC:
	"""
	The PC algorithm for structure learning. It applies independence test to sets of variables to recover the strucutre of the Bayesian Network	
	"""

	def __init__(self, n_adjacent=1, itest=chi2_ci_test(), itestDir=chi2_ci_test(), column_order="original", copy_data=True ):

		#self.root = root
		#self.heuristic = heuristic
		#self.smooth = smooth

		if(n_adjacent < 1):
			raise NameError("Error!!, n_adjacent has to be greater or equal than 1")

		self.n_adjacent = n_adjacent
		
		if( not ( "test" in dir(itest) )  ):
			raise NameError("ERROR: you has to provide a valid object (with the method 'test')!")
		self.itest = itest
		self.itestDir = itestDir # independence test for give directions

		self.column_order = column_order
		self.copy_data = copy_data

		self.structure = [] 	# a matrix which contains the structure
		self.labels = []		# labels of the matrix ## quiza no es necesario

		self.sep = []	# based in "Causal inference and causal explanation with background knowledge" of Christopher Meek, 2.1.1 Phase 1
		return

	def createStructure(self, data):			
		"""
		 data is the matrix of data, the columns are the attributes
		 root: is the index of the attribute which will be selected as the root of the tree (can be an int random value)
		 I: can be utils.MI or utils.CMI
		 Z: is used if CMI was selected
		 heuristic: True to use the heuristic to select the root node (The parameter "root is ignored")
		"""

		shD = np.shape(data)
		if(len(shD) != 2):
			raise NameError("Error!!!, data has to be a matrix (n_instances x m_attributes) ")

		if(self.column_order != "original"):
			shco = np.shape(self.column_order)
			if( len(shco) != 1 ):
				raise NameError("Error!!!, colum_order is a list/ndarray of shape (m_attributes,)")
			else:
				if(shco[0] != shD[1] ):
					raise NameError("Error!!!, size of column_order is different to the number of attributes on data.")

			for i in range(shco[0]):
				if(not i in self.column_order):
					raise NameError("Error!!!, column ",i," was not found in column_order.")

			daux = np.zeros( shD )

			for i in range(shD[1]):
				daux[:,i] = data[:, self.column_order[i] ] 

			if(self.copy_data):
				# data is change its content
				data[:,:] = daux[:,:]
				del daux
			else:
				# data now points to daux
				data = daux	



		self.structure = np.zeros( (shD[1],shD[1]), dtype=int)+1 - np.identity(shD[1], dtype=int)
		self.sep = []

		if(self.n_adjacent > (shD[1]-1)):	# adjust the max number of parents
			MP = shD[1]
		else:
			MP = self.n_adjacent + 1
		
		for x in range(shD[1]):		#walk over the attributes
			adjX = list(np.where(self.structure[x] == 1)[0])	# nodes adjacent to X
			for y in adjX:	# nodes adjacent to X in the graph
				if(y > x):	# this in order to avoid to repeat tests
					flag = False
					adjX_Y = adjX.copy()
					adjX_Y.remove(y)	# ADJ(X) - {Y}
					for i in range(1,MP):	# size of the combination
						for s in combinations(adjX_Y,i):	# walk by the combinations of size i
							# build the conditional part, set Z
							z = np.column_stack([ data[:,s[0]] ])
							for j in range(1,i):
								z = np.column_stack([ z,data[:,s[j]] ])

							if(self.itest.test( data[:,x], data[:,y], z ) ):
								# True, X and Y are independent given Z, 
								#		so, the arc X - Y, is removed
								self.structure[x,y] = 0
								self.structure[y,x] = 0
								flag = True

								# S make independent X,Y
								self.sep.append([x,y,s])

								break # it is not necessary to evaluate the rest of combinations

						if(flag):
							break # it is not necessary to evaluate the rest of combinations

		#self.giveDirections(self.heuristic)
		self.giveDirectionItest(data)
		return self.structure.copy()


	def giveDirectionItest(self,data):
		
		shs = np.shape(self.structure)

		for i in range(shs[0]):		# X
			for j in np.where(self.structure[i])[0]:	# Z
				if(self.structure[j,i]):	# if the edge is undirected 
					#flag = False
					for k in np.where(self.structure[j])[0]:	# Y
						#print("st: ",self.structure[k,j])
						if(self.structure[k,j] and (k > i)):	# if the edge is undirected, and do not return to X (X-Z-X)
							if( (self.structure[i,k] or self.structure[k,i]) ):
								# There is direction between X-Y, so skip
								continue
							else:
								# then apply the conditional test I(X,Y | Z)					
								if(not self.itestDir.test(data[:,i],data[:,k], np.column_stack([ data[:,j] ]) ) ):
									#if there is no independence given Z, 
									#	then, it orients the the edges creating a V-structure
									#		X -> Z <- Y
									#print("******************direction, ",i,j,k)
									self.structure[j,i] = 0
									self.structure[j,k] = 0
									#flag =True		

									# the edge X-Z, now has direction, we have to advance to the next branch
									break
								#else:
									#print("****************** no, ",i,j,k)


	def orientationRules(self):
		"""
		This function applies the patterns desbribed by Christopher Meek in "Causal inference and causal explanation with background knowledge"

		"""

		def isUndirected(structure, row, col):
			#return True if the the edge row-col is undirected
			return (structure[row,col] and structure[col,row])	

		def R3(structure,row,col, p_row):
			# p_row is parent of row
			# return True, if a edge received direction
			#shs = np.shape(structure)

			for k in np.where(structure[:,row])[0]:
				#********************dudas en el k>p_row (quiza eliminar)
				#if( k > p_row and (not structure[row,k] )):		# if k is parent of row(i)
				#print("struc-er: ",structure[row,k])
				if((not structure[row,k] ) and k != p_row):		# if k is parent of row(i)
					if(structure[p_row,k] or structure[k,p_row] ):
						#there is no pattern, when edge k <--> ii
						return False
					else:
						if(structure[k,col] and structure[col,k]):	# if edge k -- j is undirected; j is col
							# assign direction
							#print("R3: ",row,col,p_row,k)
							structure[i,col] = 0	# delete i -> j			
							return True
			return False

		def R4(structure,row,col, p_row):
			# p_row is parent of row
			# return True, if a edge received direction
			#shs = np.shape(structure)

			for k in np.where(structure[row])[0]:
				if( (not structure[k,row] )):		# if k is child of row(i)
					if(structure[p_row,k] or structure[k,p_row] ):
						#there is no pattern, when edge k <--> ii
						return False
					else:
						if(isUndirected(structure,k,col)):	# if edge k -- j is undirected; j is col
							# assign direction
							#print("R4: ",row,col,p_row,k)
							structure[k,col] = 0	# delete k -> j			
							return True
			return False
	
		shs = np.shape(self.structure)

		cst = self.structure.copy()

		while(True):
			# first checks for R3 and R4 patters
			for i in range(shs[0]):	#walk over rows
				for j in range(shs[1]):	#walk over cols
					if( isUndirected(self.structure,i,j) ):
						# i-j edge is undirected
						for ii in np.where(self.structure[:,i])[0]:	# parents of i
							if(self.structure[i,ii]):
								# edge i -- ii is undirected
								continue

							if(self.structure[ii,j]):	
								if(self.structure[j,ii]):	# ii -- j, undirected
									########################################################
									#	implment here R3, and R4
									########################################################
									if( R3(self.structure,i,j,ii) ):
										break
									else:
										if(R4(self.structure,i,j,ii)):
											break

								else:	# ii-> j, ii-> i
									# there is no pattern
									continue
							#else:
							#	if(self.structure[j,ii]):	# ii <- j
							#		# APPLY R2
							#		print("R2: ",i,j,ii)
							#		self.structure[i,j] = 0	# delete i -> j							
							#	else:	# ii  j, there is no link
							#		# APPLY R1
							#		print("R1: ",i,j,ii)
							#		self.structure[j,i] = 0	# delete j -> i

							#	break	# break the cycle, cause the edge i -- j now has direction

			#then checks for patters R2 and R1
			for i in range(shs[0]):	#walk over rows
				for j in range(shs[1]):	#walk over cols
					if( isUndirected(self.structure,i,j) ):
						# i-j edge is undirected
						for ii in np.where(self.structure[:,i])[0]:	# parents of i
							if(self.structure[i,ii]):
								# edge i -- ii is undirected
								continue

							if(self.structure[ii,j]):	
								#if(self.structure[j,ii]):	# ii -- j, undirected
								#	########################################################
								#	#	implment here R3, and R4
								#	########################################################
								#	if( R3(self.structure,i,j,ii) ):
								#		break
								#	else:
								#		if(R4(self.structure,i,j,ii)):
								#			break

								#else:	# ii-> j, ii-> i
								#	# there is no pattern
								#	continue
								continue
							else:
								if(self.structure[j,ii]):	# ii <- j
									# APPLY R2
									#print("R2: ",i,j,ii)
									self.structure[i,j] = 0	# delete i -> j							
								else:	# ii  j, there is no link
									# APPLY R1
									#print("R1: ",i,j,ii)
									self.structure[j,i] = 0	# delete j -> i

								break	# break the cycle, cause the edge i -- j now has direction

			if(np.all(cst == self.structure)):
				#the structure cannot be improved, so break the cycle
				break
			else:
				cst = self.structure.copy()

	def orientationRules2(self):
		"""
		This function applies the rules for patterns desbribed by Christopher Meek in "Causal inference and causal explanation with background knowledge"

		"""

		def isUndirected(structure, row, col):
			#return True if the the edge row-col is undirected
			return (structure[row,col] and structure[col,row])	

		def R3(structure,row,col, p_row):
			# p_row is parent of row
			# return True, if a edge received direction
			#shs = np.shape(structure)

			for k in np.where(structure[:,row])[0]:
				#********************dudas en el k>p_row (quiza eliminar)
				#if( k > p_row and (not structure[row,k] )):		# if k is parent of row(i)
				#print("struc-er: ",structure[row,k])
				if((not structure[row,k] ) and k != p_row):		# if k is parent of row(i)
					if(structure[p_row,k] or structure[k,p_row] ):
						#there is no pattern, when edge k <--> ii
						return False
					else:
						if(structure[k,col] and structure[col,k]):	# if edge k -- j is undirected; j is col
							# assign direction
							#print("R3: ",row,col,p_row,k)
							structure[i,col] = 0	# delete i -> j			

							if( gc.isThereCycleU_D_G(structure) ):	
								# if the new directed edge generates a cycle, then it get back to undirected
								#print("del R3: ",row,col,p_row,k)
								structure[i,col] = 1	# 
							else:		
								return True

			return False

		def R4(structure,row,col, p_row):
			# p_row is parent of row
			# return True, if a edge received direction
			#shs = np.shape(structure)

			for k in np.where(structure[row])[0]:
				if( (not structure[k,row] )):		# if k is child of row(i)
					if(structure[p_row,k] or structure[k,p_row] ):
						#there is no pattern, when edge k <--> ii
						return False
					else:
						if(isUndirected(structure,k,col)):	# if edge k -- j is undirected; j is col
							# assign direction
							#print("R4: ",row,col,p_row,k)
							structure[k,col] = 0	# delete k -> j	

							if( gc.isThereCycleU_D_G(structure) ):	
								# if the new directed edge generates a cycle, then it get back to undirected
								#print("del R4: ",row,col,p_row,k)
								structure[k,col] = 1	# 
							else:		
								return True
			return False
	
		shs = np.shape(self.structure)

		cst = self.structure.copy()

		while(True):
			# first checks for R3 and R4 patters
			for i in range(shs[0]):	#walk over rows
				for j in range(shs[1]):	#walk over cols
					if( isUndirected(self.structure,i,j) ):
						# i-j edge is undirected
						for ii in np.where(self.structure[:,i])[0]:	# parents of i
							if(self.structure[i,ii]):
								# edge i -- ii is undirected
								continue

							if(self.structure[ii,j]):	
								if(self.structure[j,ii]):	# ii -- j, undirected
									########################################################
									#	implment here R3, and R4
									########################################################
									if( R3(self.structure,i,j,ii) ):
										break
									else:
										if(R4(self.structure,i,j,ii)):
											break

								else:	# ii-> j, ii-> i
									# there is no pattern
									continue
							else:
								if(self.structure[j,ii]):	# ii <- j
									# APPLY R2
									#print("R2: ",i,j,ii)
									self.structure[i,j] = 0	# delete i -> j																

									if( gc.isThereCycleU_D_G(self.structure) ):	
										# if the new directed edge generates a cycle, then it get back to undirected
										#print("del R2: ",i,j,ii)
										self.structure[i,j] = 1	# 
									else:
										break	# break the cycle, cause the edge i -- j now has direction

								else:	# ii  j, there is no link
									# APPLY R1
									#print("R1: ",i,j,ii)
									self.structure[j,i] = 0	# delete j -> i

									if( gc.isThereCycleU_D_G(self.structure) ):	
										# if the new directed edge generates a cycle, then it get back to undirected
										#print("del R1: ",i,j,ii)
										self.structure[j,i] = 1	# 
									else:
										break	# break the cycle, cause the edge i -- j now has direction


			if(np.all(cst == self.structure)):
				#the structure cannot be improved, so break the cycle
				break
			else:
				cst = self.structure.copy()


			

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
				


