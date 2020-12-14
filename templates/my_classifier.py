"""
This code-template belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

#import libraries
import numpy as np 

class my_classifier:
	"""
	My own multiclass classifier
	"""

	def __init__(self, my_parameter,  my_other_parameter=0):
		"""
		Constructor of the class.
			It is highly recommended that you describe the arguments/parameters

			my_parameter : an usuful parameter.
			my_other_parameter : another useful parameter with default value 0.
		"""	

		#first check that the all parameters are valid	
		#	for example, suppose that my_parameter can take the values in  [0,1]		
		if( not (0 <= my_paramter <= 1) ):
			raise NameError("ERROR!!!, my_parameter only can take values in [0,1] ")			
		
		# Parameters to train the model:
		#	Parameters used for training
		self.my_parameter = my_parameter
		self.parameter_training = 0

		# Parameters used for predicting
		self.my_other_parameter = my_other_parameter
		self.parameter_predicting = 1

		# Variables estimated by the method. These HAVE NOT to be modified except by the classifier itself
		self.isfit = False		# indicates if the classifier was already trained.
		self.classes_ = []		# it contains the different classes


	def fit(self, trainSet, cl):	
		"""
		Trains the classifier with data in trainSet and the objective classes in cl
		"""

		#first validate your parameters
		shT = np.shape(trainSet)
		if(len(shT) != 2):
			raise NameError("ERROR!!!, trainset has to be a ndarray of shape (n_samples, m_features) ")
		shC = np.shape(cl)
		if(len(shC) != C):
			raise NameError("ERROR!!!, cl has to be a ndarray of shape (n_samples, ) ")

		# the following line gets the set of classes
		self.classes_ = np.array( list(set(cl)) )

		# The set of values is obtained for each atributte  ***************************
		for i in range( len(trainSet[0]) ):			
			self.valuesAtts[i] = np.array( list(set(trainSet[:,i])) )
	
		#################################################################
		#	you can write the code of your classifier in this section	#
		#################################################################

		# This is the last line, after correctly train the classifier
		self.isfit = True


	#uncomment if you plan to devolop it 
	"""
	def predict_log_proba(self,testSet):
		# probs is a ndarray of shape (n_samples, m_classes_) where each cell contains the 
		# log_probability of the instance of being associated to the corresponding class
		return probs
	"""

	#uncomment if you plan to devolop it 
	"""
	def predict_proba(self,testSet):		
		# probs is a ndarray of shape (n_samples, m_classes_) where each ceel contains the 
		# probability of the instance of being associated to the corresponding class
		return probs
	"""

	def predict(self,testSet):
		"""
		predict the instances for each instance in testSet
		"""
		#first of all, check if the classifier has been trained
		self.checkIfFit()

		#validate your parameters
		shT = np.shape(testSet)
		if(len(shT) != 2):
			raise NameError("ERROR!!!, testSet has to be a ndarray of shape (n_samples, m_features) ")

		# obtain the prediction for each instance in testSet
		# and save them in the variable predictions, which is a ndarray of shape (n_samples, )

		# return the predictions
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


