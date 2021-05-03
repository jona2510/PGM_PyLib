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

"""
Probabilities of A given its parents (any number of parents)
 P(A | Pa(A) )
"""
class probsND:

	# parents is an array which contains the number of values that can take each one
	# the first is A, the last is the Class, the others are the parents of A
	# positions has the original position of the variables/parents in the structure
	def __init__(self, variables, positions, smooth = 0.1):	
		# all this variable has to contain the information of the class
		#    that is, the class mus be seen as another attribute
		#print("variables:")
		#print(variables)
		z= [ len(variables[positions[i]]) for i in range(len(positions)) ]
		#print("z: ")
		#print(z)


		self.probabilities = np.zeros(z)	#np.zeros(parents)	# creates automatically the n-dimentional array		# P( A | Pa(a))
		self.variables = variables.copy()		#has the number of values that each variable can take, receive the full dictionary
		self.positions = positions.copy() #np.zeros( len(parents) ).astype(int)	#it helps to identify the position of the variable in the orginal structure (data)
		self.smooth = smooth

	#data contains the whole data, and at the end  the column of class is concatenated
	def estimateProbs(self,data):
		position = np.zeros( len(self.positions) ).astype(int)

		# counting
		for i in range(len(self.variables[ self.positions[0] ])):
			position[0] = i 
			x = set( np.where(data[:, self.positions[0] ] == self.variables[ self.positions[0] ][i] )[0] )

			if( len(self.positions) > 1 ):
				self.recCount(1, data, x, position)
			

		# estimate probabilities P( A | Pa(A) )
		position = np.zeros( len(self.positions) ).astype(int)

		if(len(self.positions) > 1):
		
			for i in range(len(self.variables[ self.positions[1] ])):	# it begin in the position 1 of the dic, that is, the first parent 
				position[1] = i
				self.recProbs( 1, position) 

		else:
			self.probabilities = self.probabilities / sum( self.probabilities )


	def recProbs(self, index, position):

		if(index < len(self.positions)):	
			for i in range( len(self.variables[ self.positions[index] ]) ):
				position[index] = i
				self.recProbs(index + 1, position)
		else:	#estimate the probabilities, if all the parents have been visited
			s = 0.0
			for i in range( len(self.variables[ self.positions[0] ]) ):
				position[0] = i
				s += self.probabilities[ tuple(position) ]

			for i in range( len(self.variables[ self.positions[0] ]) ):
				position[0] = i
				self.probabilities[ tuple(position) ] = self.probabilities[ tuple(position) ] / s


	def recCount(self, index, data, set_, position):

		for i in range( len(self.variables[ self.positions[index] ]) ):
			position[index] = i 
			if(set_ != set()):
				x = set( np.where(data[:, self.positions[index] ] == self.variables[ self.positions[index] ][i])[0] ) & set_
			else:
				x=set()

			if( len(self.positions) > (index + 1) ):
				self.recCount(index + 1, data, x, position)
			else:
				#it assigns the value
				self.probabilities[ tuple(position) ] = len(x) + self.smooth
		
	# the full instance and at the end concatenated with the class
	def probsInstance(self, instance):	#it return the "probabilities" for each class
		
		p = [ np.where( self.variables[ self.positions[i] ] == instance[ self.positions[i] ] )[0][0] for i in range( len(self.positions) ) ]
		# It must access directly to the probabilitie
		pr = self.probabilities[tuple(p)]
		#print("---Prob: " + str(pr))
		return pr 

			
		

	# ***************** Below is not useful anymore
	
	# vector constains the position of the cell of interest
	# vector is an array of size len(variables)
	def posVector(self, vector):
		pos = 0
		pos2 = 0 
		jump = 1	#the jump

		#deberia ser un for, contando desde el centro hacia afuera
		for i in range( len(self.variables) - 1, -1, -1 ):
			
			pos += vector
			

		for i in len(self.variables):
			pos +=  self.variables[i] * vector[i]
	
		return pos


class augmentedBC:


	"""
	Constructor of the class
		algStructure:	Algorithm used to generate the structure. Available algs. : CLprocedureCMI 	
		smooth 		: 	it is used to smooth the probabilities (a number greater than 0)
		usePrior	: 	in the prediction phase, if True then considers the prior probabilities, that is, 
						the probability of the class, if False then they are not considered
		meta		:	could be avoided

	"""
	def __init__(self, algStructure="auto", smooth=0.1, usePrior=True, meta=""):
		# Parameter to train the model:
		#	Parameters used for training
		self.algStructure = algStructure
		self.smooth = smooth	
		self.meta = meta


		if(self.smooth < 0):
			raise NameError("smooth has to be greater or equal than 0")			


		#	Parameters used for predicting
		self.usePrior = usePrior

		
		# Variables estimated by the method. These HAVE NOT to be modified except by the classifier itself

		self.isfit = False
		self.structure = [] 	# a matrix which contains the structure
		self.labels = []		# labels of the matrix ## quiza no es necesario

		self.classes_ = []		# it contains the different classes
		self.probsClasses = []	# it contains the probabilities of each class: P(C)

		self.probsAtts = {}		# it contains the probabilities of each attribute: P(Ai | C)
		self.valuesAtts = {}	# it contains the values that the attributte can take

	
	# it trains the classifier with data (train) and use the meta-data
	# cl is the class to which every instance is associated
	# meta could be avoided
	def fit(self, trainSet, cl):

		shTr = np.shape(trainSet)	# shape of the train set
		if(len(shTr) != 2):
			raise NameError("The train set has to be a numpy two-dimension matrix (intances x attributtes)")			

		if(len(trainSet) != len(cl)):
			raise NameError("The number of elements in data is different from classes (cl)")

		n=shTr[1]	#number of atts

		self.structure = np.zeros((n+1,n+1),dtype=int)	

		# all the attributes are descendents of the class
		self.structure[-1] = 1
		self.structure[-1, -1] = 0	# the class is not descendent of itself

		# obtain the structure of the 
		sh = np.shape(self.algStructure)	# shape of structure, it is useful if the structure is given in algStructure

		if(self.algStructure == "auto"):
			#st = trees.CLprocedure()
			#self.structure[0:-1,0:-1] = st.createTree(trainSet, I="CMI", Z=cl, heuristic=True)
			st = trees.CLP_CMI()	# default parameters for CLP_CMI
			self.structure[0:-1,0:-1] = st.createStructure(trainSet, cl)
		elif( len(sh) == 2 ):
			if( sh[0] == n == sh[1] ):
				self.structure[0:-1,0:-1] = self.algStructure
			elif( sh[0] == (n+1) == sh[1] ):	# in the case a matrix with (n+1, n+1) is provided, where de last row/column correspond to the class
				self.structure[:,:] = self.algStructure[:,:]
			else:
				raise NameError("The provided structure has to be a matrix with size (n x n) or (n+1,n+1), where n is the number of attributes")
		else:			
			raise NameError("algStructure only can take the values: 'auto' or a matrix with size (n x n), where n is the number of attributes")


		#print("Structure:")
		#print(self.structure)
		# classes ******************************

		self.classes_ = np.array( list(set(cl)) )
		self.probsClasses = np.zeros(len(self.classes_))		

		
		#with smooth
		for i in range( len(self.probsClasses) ):
			self.probsClasses[i] = (np.sum( cl == self.classes_[i] ) + self.smooth)/ (len(cl) + self.smooth*len(self.classes_))
			
		#print("classes:")
		#print(self.classes_)
		#print()
		#print("probabities of classes:")
		#print(self.probsClasses)


		# attributes **************************

		if( (self.meta != "") and (len(self.meta) == len(trainSet[0])) ):
			self.valuesAtts = self.meta.copy()
			#print("WARNING: The values that each attribute can take are given as input!")
		else:
			for i in range( len(trainSet[0]) ):
				self.valuesAtts[i] = np.array( list(set(trainSet[:,i])) )

		valuesAux = self.valuesAtts.copy()
		valuesAux[ len(valuesAux) ] = self.classes_
		#print("valuesAux:")
		#print(valuesAux)
		
		data = np.column_stack( [trainSet, cl] )

		for i in range( len(trainSet[0]) ):			
			a = int(np.sum( self.structure[:,i] ))	#number of parents, included class
			positions = np.zeros( a + 1 ).astype(int)	# plus one
			positions[0] = i
			positions[1:] = np.where( self.structure[:,i] == 1 )[0]
			self.probsAtts[i] = probsND(valuesAux, positions, self.smooth )	#np.zeros( ( len(self.valuesAtts[i]) , len(self.classes_) ) )

			self.probsAtts[i].estimateProbs(data)

			#print("probs de "+str(i)+":")
			#print(self.probsAtts[i].probabilities)

		self.isfit = True



	def predict_log_proba(self,testSet):
		# it is supposed that the attributtes of all all the instances are known, 
		#	that is, the are not missing values

		# we are using the variant of sum of logarithms instead of multiplication of the probabilities
		#		see Sucar, L. E., Probabilistic Graphical Models, Chapter 4: Bayesian Classifiers

		self.checkIfFit()	# first check if the classifier is already trained

		#Verifying that the number of attributes of testSet correspond to the number of attributes of the classifier
		if( len(testSet[0]) != len(self.valuesAtts) ):
			raise NameError("The number of attributes does NOT correspond, provided: "+str(len(testSet[0])) + ", it has to be: " +str(len(self.valuesAtts)) )
	

		probs = np.zeros(( len(testSet), len(self.classes_) ))	#it contains the probability of belonging to each class


		if(self.usePrior):
			#walk over the instances
			for i in range(len(testSet)):
				#walk over the classes
				for k in range(len(self.classes_)):
					probs[i][k] = np.log( self.probsClasses[k] )
	

		#walk over the instances
		for i in range(len(testSet)):
			#walk over the classes
			for k in range(len(self.classes_)):
				#walk over the attributes
				for j in range(len(self.valuesAtts)):
					#print("testSet, cl")
					#print(testSet[i])
					#print(self.classes_[k])
					#print(self.classes_)
					#print("+++++++++++")
					x = np.concatenate( [testSet[i],  [ self.classes_[k] ] ] )
					probs[i][k] += np.log( self.probsAtts[j].probsInstance( x ) ) 

		#print("log Probs:")
		#print(probs)
		return probs

	def predict_proba(self,testSet):		
		#First apply exp() to the scores obtained by predict_log_proba()
		probs = np.exp( self.predict_log_proba(testSet).copy() )
		s = np.sum(probs,axis=1)

		#nortmalize the probabilities
		probs = probs / s[:,None]

		return probs


	def predict(self,testSet):
		probs = self.predict_log_proba(testSet)

		#predictions=np.zeros(len(testSet)).astype(str)
		predictions = []

		for i in range(len(testSet)):
			posA = np.where( probs[i] == max(probs[i]) )[0][0]
			#predictions[i] = self.classes_[ posA ]
			predictions.append( self.classes_[ posA ] )

		return predictions

	def exactMatch(self, real, prediction):
		if(len(real) != len(prediction)):
			raise NameError("The size of real and predictions are differents")

		em=0.0
		for i in range(len(real)):
			if(real[i] == prediction[i]):
				em += 1.0

		em /= len(real)

		return em



	def checkIfFit(self):
		"""
		Check is the classifiers is already trained, 
		if it is not, then raises a exeption
		"""
		if(not self.isfit):
			raise NameError("Error!: First you have to train ('fit') the classifier!!")










