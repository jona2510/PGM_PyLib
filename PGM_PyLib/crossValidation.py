"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

# this cv works for multiclass problems
import numpy as np

class crossValidation():

	def __init__(self, dataset, clSet, folds = 5, stratified = True):

		self.dataset = dataset	# copy ?
		self.clSet = clSet		# set of classes
		self.folds = folds
		self.stratified = stratified

		self.checkErrors()

		
		self.classes = list( set(clSet) )

		flag = False	# 

		self.dPos = {}	# dictionary with the position of the instances for each class
		for x in self.classes:
			self.dPos[x] = np.where( self.clSet == x )[0]
			nl = len(self.dPos[x])
			if( nl < self.folds ):
				print("WARNING: class " + str(x) + " has " + str(nl) + " elements, which have to be distributed in " + str(self.folds) + " folds")

		#self.valuesAtts = []	# it contains the values that the attributte can take
		#for i in range( len(self.dataset[0]) ):			
		#	self.valuesAtts.append( np.array( list(set(self.dataset[:,i])) ) )

		self.valuesAtts = {}	# it contains the values that the attributte can take
		for i in range( len(self.dataset[0]) ):			
			self.valuesAtts[i] = np.array( list(set(self.dataset[:,i])) ) 

	def getValuesAtts(self):
		"""
		Return the ordered-list with the values that each attribute can take
		"""
		return self.valuesAtts.copy()

	def getClasses(self):
		"""
		Return the values that the variable-class can take
		"""
		return self.classes.copy()
		
	def checkErrors(self):
		if (self.folds <= 1):
			raise NameError("The number of folds has to be greater or equal than 2")

		if(type(self.dataset) is not np.ndarray ):
			raise NameError("dataset has to be a two-dimension numpy matrix, you could use numpy.array(data) to transform your data ")
	
		if(type(self.clSet) is not np.ndarray ):
			raise NameError("classes has to be a numpy vector, you could use numpy.array(data) to transform your data")

		shD = np.shape(self.dataset)
		shC = np.shape(self.clSet)
		
		if(len(shD) != 2):
			raise NameError("dataset has to be a two-dimension numpy matrix")

		if(len(shC) != 1):
			raise NameError("classes has to be a numpy vector")

		if(shD[0] != shC[0]):
			raise NameError("the size of classes if different to the number of instances")
	

	def positionsOfFold(self, fold):
		if( (fold < 0) or (fold >= self.folds) ):
			raise NameError("Fold only can take values from 0 to " + str(self.folds - 1))

		inst = np.array([]).astype(int)

		for x in self.classes:
			nl = float(len(self.dPos[x]))
			r = nl / self.folds 	#range
		
			start = int( round(fold * r) )
			end = int( round( (fold + 1) * r ) )

			inst = np.concatenate( [inst, self.dPos[x][start:end] ] )

		inst = sorted(inst)

		return inst


	def getFold(self, fold):
		pos = self.positionsOfFold(fold)

		d = set([i for i in range(len(self.dataset))])
		pos = list( d - set(pos) )
		
		ds= np.delete( self.dataset, pos, axis=0 )
		dc= np.delete( self.clSet, pos )

		return (ds,dc)
		

	def getComplementOfFold(self, fold):
		pos = self.positionsOfFold(fold)

		ds= np.delete( self.dataset, pos, axis=0 )
		dc= np.delete( self.clSet, pos )

		return (ds,dc)

	def getPercentage(self, p):
		x = 0

	def getComplementOfPercentage(self, p):
		x = 0
