"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

#libraries
import numpy as np 


class naiveBayes:

	"""
	Constructor of the class.
		smooth : it is used to smooth the probabilities (a number greater than 0)
		usePrior : in the prediction phase, if True then considers the prior probabilities, that is, 
			the probability of the class, if False then they are not considered
		meta : it can be used to provide the values that the attributes can take. e.g.
			meta={0: ['a', 'b', 'c'], 1: ['1', '2']}
			Note that the keys are numbers, which began in 0
	"""	
	def __init__(self, smooth=0.1, usePrior=True, meta=""):
		# Parameter to train the model:
		#	Parameters used for training
		self.smooth = smooth	
		self.meta = meta


		if(self.smooth < 0):
			raise NameError("smooth has to be greater or equal than 0")			


		# Parameters used for predicting
		self.usePrior = usePrior

		
		# Variables estimated by the method. These HAVE NOT to be modified except by the classifier itself

		self.classes_ = []		# it contains the different classes
		self.probsClasses = []	# it contains the probabilities of each class: P(C)

		self.probsAtts = {}		# it contains the probabilities of each attribute: P(Ai | C)
		self.valuesAtts = {}	# it contains the values that the attributte can take
		return

	# it trains the classifier with data (train) and use the meta-data
	# cl is the class to which every instance is associated
	# meta could be avoided
	def fit(self, trainSet, cl):	
		#deberia de verificar todos los parametros....
		if(self.smooth < 0):
			raise NameError("smooth has to be greater or equal than 0")			

		#print("train set:")
		#print(trainSet)
		#print("classes associated to instances:")
		#print(cl)

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


		# atributtes ***************************

		#print("train set atts"+str(len(trainSet[0])))

		if( (self.meta != "") and (len(self.meta) == len(trainSet[0])) ):
			self.valuesAtts = self.meta.copy()
			#print("WARNING: The values that each attribute can take are given as input!")

		for i in range( len(trainSet[0]) ):			
			if(self.meta == ""):
				self.valuesAtts[i] = np.array( list(set(trainSet[:,i])) )
			self.probsAtts[i] = np.zeros( ( len(self.valuesAtts[i]) , len(self.classes_) ) )
		

		#print("\nvalues of attributes:")
		#print(self.valuesAtts)

		# walk over the attributes
		for i in range(len(trainSet[0])):
			# walk over the instances/objects
			#print(len(trainSet))
			for j in range(len(trainSet)):
				z= trainSet[j,i]

				posA = np.where( self.valuesAtts[i] == trainSet[j,i] )[0][0]
				posB = np.where( self.classes_ == cl[j] )[0][0]

				#print("posA: " +str(posA)+ ", posB: "+str(posB))

				self.probsAtts[i][ posA, posB ] = self.probsAtts[i][ posA, posB ] + 1
				#probsAtts[i] = np.zeros(())

		#smooth
		for i in self.probsAtts.keys():
			self.probsAtts[i] = self.probsAtts[i] + self.smooth
			# estimate the probabilities
			for j in range(len(self.probsAtts[i])):
				self.probsAtts[i][j]=self.probsAtts[i][j]/sum(self.probsAtts[i][j])


		#print("probabilities Attributes:")
		#print(self.probsAtts)
		return


	#returns de probabilities of each class	
	def predict_log_proba(self,testSet):
		# it is supposed that the attributtes of all the instances are known, 
		#	that is, the are not missing values

		# we are using the variant of sum of logarithms instead of multiplication of the probabilities
		#		see Sucar, L. E., Probabilistic Graphical Models, Chapter 4: Bayesian Classifiers

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
				#probs[i][k] = np.log( self.probsClasses[k] )
				#walk over the attributes
				for j in range(len(self.valuesAtts)):
					posA = np.where( self.valuesAtts[j] == testSet[i,j] )[0][0]
					#print("posA: "+str(posA))

					probs[i][k] += np.log( self.probsAtts[j][ posA,k ] )

					# np.log is the natural logarithm
					#paux = paux + np.log(  )
			#print(probs[i])
			#probs[i] /= sum(probs[i])	# the results obtained by the use of the log are negative value, 
			#								therefore, it cannot be normalized

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

		return np.array( predictions )

	def exactMatch(self, real, prediction):
		if(len(real) != len(prediction)):
			raise NameError("The size of real and predictions are differents")

		em=0.0
		for i in range(len(real)):
			if(real[i] == prediction[i]):
				em += 1.0

		em /= len(real)

		return em



# it is equal to naive Bayes, except that in this class, the predicitionProba method sum and normalize the probabilities
class sumNaiveBayes(naiveBayes):

	def predict_log_proba(self, testSet):
		raise NameError("Error: the function predict_log_proba() is not implemented for class sumNaiveBayes")
	
	#returns de probabilities of each class
	def predict_proba(self,testSet):
		# it is supposed that the attributtes of all all the instances are known, 
		#	that is, the are not missing values

		# we are using the variant of sum of logarithms instead of multiplication of the probabilities
		#		see Sucar, L. E., Probabilistic Graphical Models, Chapter 4: Bayesian Classifiers

		#Verifying that the number of attributes of testSet correspond to the number of attributes of the classifier
		if( len(testSet[0]) != len(self.valuesAtts) ):
			raise NameError("The number of attributes does NOT correspond, provided: "+str(len(testSet[0])) + ", it has to be: " +str(len(self.valuesAtts)) )

		probs = np.zeros(( len(testSet), len(self.classes_) ))	#it contains the probability of belonging to each class


		if(self.usePrior):
			#walk over the instances
			for i in range(len(testSet)):
				#walk over the classes
				for k in range(len(self.classes_)):
					probs[i][k] = self.probsClasses[k]


		#walk over the instances
		for i in range(len(testSet)):
			#walk over the classes
			for k in range(len(self.classes_)):
				#probs[i][k] = np.log( self.probsClasses[k] )
				#walk over the attributes
				for j in range(len(self.valuesAtts)):
					posA = np.where( self.valuesAtts[j] == testSet[i,j] )[0][0]
					#print("posA: "+str(posA))

					probs[i][k] += self.probsAtts[j][ posA,k ]

					# np.log is the natural logarithm
					#paux = paux + np.log(  )
			#print(probs[i])
			probs[i] /= sum(probs[i])	


		return probs


	def predict(self,testSet):
		probs = self.predict_proba(testSet)

		#predictions=np.zeros(len(testSet)).astype(str)
		predictions = []

		for i in range(len(testSet)):
			posA = np.where( probs[i] == max(probs[i]) )[0][0]
			#predictions[i] = self.classes_[ posA ]
			predictions.append( self.classes_[ posA ] )

		return np.array( predictions )

