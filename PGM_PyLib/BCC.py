"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

#libraries
import PGM_PyLib.structures.trees as trees
import numpy as np
import PGM_PyLib.naiveBayes as nb
import copy
from collections import deque

class BCC:
	
	def __init__(self,chainType="parents", baseClassifier=nb.naiveBayes(), structure="auto"):#, smooth=0.1, meta=""):

		self.chainType = chainType
		self.baseClassifier = baseClassifier	# base classifier, it wil be copied for each class


		#	Parameters used for predicting

		
		# Variables estimated by the method. These HAVE NOT to be modified except by the classifier itself
		self.isfit = False
		self.dCl = {}			#dictionary which contains the classifiers
		self.dChain = {}		#dictionary with the classes that form the chain for each class	(Parents, Children, Ancestors)
		self.roots = []				#list with the root nodes
		self.structure = structure	# a matrix which contains the structure
		self.orderTC = []			# list with the order in which the classifiers has to be trained (aand the for prediction)


	# train: is a matrix (n x m)  where n is the number of instances and m the number of classes
	# cl: is a matrix (n x l)  where n is the number of instances and l the number of classes
	def fit(self, trainSet, cl):			

		shTr = np.shape(trainSet)	# shape of the train set
		if(len(shTr) != 2):
			raise NameError("The train set has to be a two-dimension numpy matrix (intances x attributtes)")			

		shCl = np.shape(cl)	# shape of the classes to which the instances are associated
		if(len(shCl) != 2):
			raise NameError("The classes has to be a two-dimension numpy matrix (intances x classes)")			

		if(shTr[0] != shCl[0]):
			raise NameError("The number of elements in data is different from classes (cl)")	

		if( not (( "fit" in dir(self.baseClassifier) ) and ( "predict" in dir(self.baseClassifier) )) ):
			raise NameError("ERROR: you has to provide a valid classifier!")				

		n = shTr[1]	#number of atts
		nCl = shCl[1]	#number of classes

		shSt = np.shape(self.structure)	# shape of structure, it is useful if the structure is given in structure

		if(self.structure == "auto"):
			#st = trees.CLprocedure()
			#self.structure[0:-1,0:-1] = st.createTree(trainSet, I="CMI", Z=cl, heuristic=True)
			st = trees.CLP_MI()	# default parameters for CLP_MI
			self.structure = st.createStructure(cl)
		elif( len(shSt) == 2 ):
			if(not ( (shSt[0] == nCl) and (shSt[0] == shSt[1]) ) ):
				#self.structure[0:-1,0:-1] = self.algStructure
				#else:
				raise NameError("The provided structure has to be a matrix with size (n x n), where n is the number of classes")

			# verify that the structure doesn't have cycles

		else:			
			raise NameError("structure only can take the values: 'auto' or a matrix with size (n x n), where n is the number of classes")


		if(self.chainType == "parents"):
			self.chainParents()
		elif(self.chainType == "ancestors"):
			self.chainAncestors()
		elif(self.chainType == "children"):
			self.chainChildren()
		else:
			raise NameError("chainType only can take the values 'parents', 'ancestors' and 'children'")



		# Begin the training
		#	Following the order in self.orderTC		(in train it is not necessary)
		for x in self.orderTC:
			#copy the base classifier
			self.dCl[x] = copy.deepcopy(self.baseClassifier)

			#add the chain as extra attributes
			if( len(self.dChain) == 0 ):
				#if there are not extra attributes, the it train with the train set directly
				self.dCl[x].fit( trainSet, cl[:,x] )
			else:
				# create an auxiliar matrix for adding the extra attributes
				# I think this way of copy is expensive
				trainAux = copy.deepcopy( trainSet )
				for y in self.dChain[x]:
					trainAux = np.column_stack( [trainAux, cl[:,y]] )
				
				# train the classifier
				self.dCl[x].fit(  trainAux, cl[:,x] )
				
				del trainAux

		self.isfit = True


	def predict_log_proba(self, testSet):		
		self.checkIfFit()	# first check if the classifier is already trained
		if( "predict_log_proba" not in dir(self.baseClassifier) ):
			raise NameError("ERROR: the provided base classifier does NOT have the method 'predict_log_proba'!")
		
		# check if testSet has the correct dimensions

		pr = [ [] for x in range(len(self.structure)) ]

		pr_prob = [ [] for x in range(len(self.structure)) ]

		# Begin the prediction
		#	Folowing the order in self.orderTC
		for x in self.orderTC:			
			#copy the base classifier
			#self.dCl[x] = copy.deepcopy(self.baseClassifier)

			#add the chain as extra attributes
			if( len(self.dChain) == 0 ):
				#if there are not extra attributes, then it predicts with the test set directly
				pr[x] = self.dCl[x].predict( testSet )
				#self.dCl[x].fit( trainSet, cl[:,x] )

				# in a second step, obtain the probabilities:
				pr_prob[x] = self.dCl[x].predict_log_proba( testSet )
				
			else:
				# create an auxiliar matrix for adding the extra attributes
				# I think this way of copy is expensive
				testAux = copy.deepcopy( testSet )
				for y in self.dChain[x]:
					testAux = np.column_stack( [testAux, pr[y]] )

				# predict
				pr[x] = self.dCl[x].predict( testAux )

				# in a second step, obtain the probabilities:
				pr_prob[x] = self.dCl[x].predict_log_proba( testAux )
				
				del testAux

		##process the output
		#predictions = np.column_stack([ pr[0] ])
		#for i in range(1, len(pr)):
		#	predictions = np.column_stack( [predictions, pr[i]] )
		#
		#return predictions
		return pr_prob


	def predict_proba(self, testSet):	
		self.checkIfFit()	# first check if the classifier is already trained	
		if( "predict_proba" not in dir(self.baseClassifier) ):
			raise NameError("ERROR: the provided base classifier does NOT have the method 'predict_proba'!")
		
		# check if testSet has the correct dimensions


		pr = [ [] for x in range(len(self.structure)) ]

		pr_prob = [ [] for x in range(len(self.structure)) ]

		# Begin the prediction
		#	Folowing the order in self.orderTC
		for x in self.orderTC:			
			#copy the base classifier
			#self.dCl[x] = copy.deepcopy(self.baseClassifier)

			#add the chain as extra attributes
			if( len(self.dChain) == 0 ):
				#if there are not extra attributes, then it predicts with the test set directly
				pr[x] = self.dCl[x].predict( testSet )
				#self.dCl[x].fit( trainSet, cl[:,x] )

				# in a second step, obtain the probabilities:
				pr_prob[x] = self.dCl[x].predict_proba( testSet )
				
			else:
				# create an auxiliar matrix for adding the extra attributes
				# I think this way of copy is expensive
				testAux = copy.deepcopy( testSet )
				for y in self.dChain[x]:
					testAux = np.column_stack( [testAux, pr[y]] )

				# predict
				pr[x] = self.dCl[x].predict( testAux )

				# in a second step, obtain the probabilities:
				pr_prob[x] = self.dCl[x].predict_proba( testAux )
				
				del testAux

		##process the output
		#predictions = np.column_stack([ pr[0] ])
		#for i in range(1, len(pr)):
		#	predictions = np.column_stack( [predictions, pr[i]] )
		#
		#return predictions
		return pr_prob


	def predict(self, testSet):
		self.checkIfFit()	# first check if the classifier is already trained
		# check if testSet has the correct dimensions

		pr = [ [] for x in range(len(self.structure)) ]

		# Begin the prediction
		#	Folowing the order in self.orderTC
		for x in self.orderTC:			
			#copy the base classifier
			#self.dCl[x] = copy.deepcopy(self.baseClassifier)

			#add the chain as extra attributes
			if( len(self.dChain) == 0 ):
				#if there are not extra attributes, then it predicts with the test set directly
				pr[x] = self.dCl[x].predict( testSet )
				#self.dCl[x].fit( trainSet, cl[:,x] )
				
			else:
				# create an auxiliar matrix for adding the extra attributes
				# I think this way of copy is expensive
				testAux = copy.deepcopy( testSet )
				for y in self.dChain[x]:
					testAux = np.column_stack( [testAux, pr[y]] )
				
				# train the classifier
				pr[x] = self.dCl[x].predict( testAux )
				
				del testAux

		#process the output
		predictions = np.column_stack([ pr[0] ])
		for i in range(1, len(pr)):
			predictions = np.column_stack( [predictions, pr[i]] )

		return predictions


	def chainParents(self):
		#roots = []
		queue = deque()	# the queue
		visited = []	# list with the visited nodes
		#visited = np.zeros(len(self.structure)).astype(bool)
		

		for i in range(len(self.structure)):
			if( np.sum(self.structure[:,i]) == 0 ):	# the coulums with zeros indicate that they are roots
				queue.append(i)
			#this gets the parents of the i-th class, which form the chain of the i-th node/class		
			self.dChain[i] = set( np.where( self.structure[:,i] == 1 )[0] )	# a set is better option

		# generate the order for training the classifiers, and then predict
		while( len(queue) > 0 ):
			y = queue.popleft()		#get the node/class
		
			flag = True
			#firts checks if all its parents are already visited
			for x in self.dChain[y]:
				if( x not in visited ):	
					flag = False
					break

			if(flag):
				#if all the parents are visited
				# the node/class is visited
				visited.append(y)
				# and add its children at the end of the queue	(only those that haven't been visited and aren't in the queue)
				children = np.where( self.structure[y] == 1 )[0]
				
				for x in children:
					if( (x not in visited) and (x not in queue) ):
						queue.append(x)				
			else:				
				#if SOME parent has NOT been visited
				# insert the node/class in the end of the queue 
				queue.append(y)	

		if( ( len(set(visited)) != len(self.structure) ) or ( len(visited) != len(self.structure) )):
			raise NameError("ERROR in the construction of the order for trainig the classifiers (review code)")

		# the order to build the classifier is saved in visited
		self.orderTC = visited


	def chainAncestors(self):
		# It is executed 'chainParents()'
		self.chainParents()
		
		# But in self.dChain has to be added all the ancestors of each class/node
		for i in range(len(self.structure)):
			self.dChain[i] = self.getAncestors(i)


	def chainChildren(self):
		#roots = []
		queue = deque()	# the queue
		visited = []	# list with the visited nodes
		#visited = np.zeros(len(self.structure)).astype(bool)
		

		for i in range(len(self.structure)):
			if( np.sum(self.structure[i]) == 0 ):	# the rows with zeros indicate that they are leaves
				queue.append(i)
			#this gets the children of the i-th class, which form the chain of the i-th node/class		
			self.dChain[i] = set( np.where( self.structure[i] == 1 )[0] )	# a set is better option

		# generate the order for training the classifiers, and then predict
		while( len(queue) > 0 ):
			y = queue.popleft()		#get the node/class
		
			flag = True
			#firts checks if all its children are already visited
			for x in self.dChain[y]:
				if( x not in visited ):	
					flag = False
					break

			if(flag):
				#if all the children are visited
				# the node/class is visited
				visited.append(y)
				# and add its parents at the end of the queue	(only those that haven't been visited and aren't in the queue)
				parents = np.where( self.structure[:,y] == 1 )[0]
				
				for x in parents:
					if( (x not in visited) and (x not in queue) ):
						queue.append(x)				
			else:				
				#if SOME child has NOT been visited
				# insert the node/class in the end of the queue 
				queue.append(y)	

		if( ( len(set(visited)) != len(self.structure) ) or ( len(visited) != len(self.structure) )):
			raise NameError("ERROR in the construction of the order for trainig the classifiers (review code)")

		# the order to build the classifier is saved in visited
		self.orderTC = visited



	# it returns the ancestors of 'index'
	def getAncestors(self, index):

		parents = set( np.where( self.structure[:,index] == 1 )[0] )
		anc = copy.copy(parents)
		
		for x in parents:
			anc = anc | self.getAncestors(x)

		return anc
		

	# this version receives matrices (real/prediction_value x classes)
	def exactMatch(self, real, prediction):

		shR = np.shape(real)
		shP = np.shape(prediction)

		if(len(shR) != 2):
			raise NameError("The dimensions of 'real' are incorrect (it has to be a two dimension matrix)")

		if(len(shP) != 2):
			raise NameError("The dimensions of 'predictions' are incorrect (it has to be a two dimension matrix)")

		#if(len(real) != len(prediction)):
		if(shR != shP):
			raise NameError("The size of real and predictions are differents")

		em=0.0
		for i in range(shR[0]):
			for j in range(shR[1]):
				if(real[i,j] == prediction[i,j]):
					em += 1.0

		em /= (shR[0] * shR[1])

		return em


	def getClasses(self):
		"""
		return a dictionary with the classes that each variable class can take,
			The key of each item is the position of the class
		Note: the classes are in clasifier.classes
		"""
		classesMD = {}
		# despues de que se haya entrenado
		for i in range(len(self.dCl)):
			classesMD[i] = self.dCl[i].classes.copy()

		return classesMD
			

	def getClasses_(self):
		"""
		return a dictionary with the classes that each variable class can take,
			The key of each item is the position of the class
		Note: the classes are in clasifier.classes_
		"""
		classesMD = {}
		# despues de que se haya entrenado
		for i in range(len(self.dCl)):
			classesMD[i] = self.dCl[i].classes_.copy()

		return classesMD


	def checkIfFit(self):
		"""
		Check is the classifiers is already trained, 
		if it is not, then raises a exeption
		"""
		if(not self.isfit):
			raise NameError("Error!: First you have to train ('fit') the classifier!!")
































