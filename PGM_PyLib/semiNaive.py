"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

#libraries
import numpy as np 
import PGM_PyLib.utils as utils	#checar directorios
import PGM_PyLib.naiveBayes as nb

# based in Miriam Martinez Arrollo's algoritmh from phd thesis "Aprendizaje de Clasificadores Beyesianos Estaticos y Dinamicos", page 51
# 	also some notation is taken from Sucar's book, "Probabilistic Graphical Models", page 50

class semiNaive:


	"""
	Constructor of the class
		validation : is the percentage of the training test that will be used to train the internal classifier, and (1-validation) will be used to evaluated the internal classifier
		epsilon	: 	is the threshold for the MI (attribute and class)
		omega 	: 	is the threshold for the CMI (two attributes given the class)		
		smooth 	: 	it is used to smooth the probabilities (a number greater than 0)
		nameAtts:	List of the name of the attributes (if "auto" the name is the index of the attribute)
		usePrior: 	in the prediction phase, if True then considers the prior probabilities, that is, 
						the probability of the class, if False then they are not considered
		meta	:	could be avoided

	"""
	def __init__(self, validation=0.8, epsilon=0.1, omega=0.1, nameAtts="auto", smooth=0.1, usePrior=True, meta=""):
		# Parameter to train the model:
		#	Parameters used for training		
		self.validation = validation
		self.epsilon = epsilon
		self.omega = omega
		self.smooth = smooth	
		self.nameAtts = nameAtts
		self.meta = meta


		if(self.smooth < 0):
			raise NameError("smooth has to be greater or equal than 0")

		if( (self.validation < 0.1) or (self.validation > 0.9) ):
			raise NameError("validation only can take values in the range: 0.1<=validation<=0.9")			


		#	Parameters used for predicting
		self.usePrior = usePrior

		
		# Variables estimated by the method. These HAVE NOT to be modified except by the classifier itself
		self.isfit = False
		self.delMI = []			# contains the index of the attributes that are deleted, # it wont be used
		self.operations = [] 	# contains the operation in order to be applied to the data (attributes), two operations: del and join
		self.opeNameAtts =[]
		self.NBC = [] 		# the Naive Bayes Classifier for the classification

		self.valuesAtts = {}	# it contains the values that the attributtes can take
		self.lvaluesAtts = {}	#lvaluesAtts.copy()
		self.orderAtts = []		# it contains the final order of the attributes
		self.trainSet = []	# data set
		self.cl = []			# classes of the dataset
		return

	# it trains the classifier with data (train) and use the meta-data
	# cl is the class to which every instance is associated
	# meta could be avoided
	# epsilon is the threshold for the MI (attribute and class)
	# omega is the threshold for the CMI (two attributes given the class)
	def fit(self, trainSet, cl):	

		if(self.smooth<0):
			raise NameError("smooth has to be greater or equal than 0")

		if( (self.validation < 0.1) or (self.validation > 0.9) ):
			raise NameError("validation only can take values in the range: 0.1<=validation<=0.9")			

		if(self.nameAtts != "auto"):
			if( len(self.nameAtts) != len(trainSet[0]) ):
				raise NameError("The number of atts is different than the number of names given")
		else:
			self.nameAtts = [ str(i) for i in range( len(trainSet[0]) ) ]

		#the first row has the name of the attributes
		self.trainSet = np.concatenate([ [self.nameAtts], trainSet ])
		self.cl = cl
	

		if( (self.meta != "") and (len(self.meta) == len(trainSet[0])) ):
			print("atts from meta")			
			for i in range( len(trainSet[0]) ):
				self.valuesAtts[ self.trainSet[0,i] ] = self.meta[i]
		else:
			# obtain some data
			for i in range( len(trainSet[0]) ):
				# avoid the first element (name of the attribute), the name of the attribute is used as key
				self.valuesAtts[ self.trainSet[0,i] ] = np.array( list(set(self.trainSet[1:,i])) )	

		# In this section the training set is divided in training_2 and validation


		# A NBC is trained, and its performance is evaluated, and it is selected as "permanent"

		self.NBC = [] 		# the Naive Bayes Classifier for the classification

		div = int( (len(self.trainSet) - 1 ) * self.validation )
		self.NBC = nb.naiveBayes(self.smooth, self.usePrior, meta = self.meta)
		#print("NBC: ")
		#print(self.NBC)
		self.NBC.fit(self.trainSet[1:(div+1)] , self.cl[:div] )#	, meta=l2valuesAtts )

		pred = self.NBC.predict( self.trainSet[(div+1):] )
		score = self.exactMatch( self.cl[div:], pred )
		#print("score NBC: "+str(score))


		# call improveStructure

		self.improveStructure( score, 1)	


		#print("number of atts: "+str(len(self.trainSet[0])))
		#print(self.trainSet[0])
		#print("Operations:")
		#print(self.operations)
		#print(self.opeNameAtts)

		#trains the classifier with all the data and the respective operations
		self.lvaluesAtts = {}	#lvaluesAtts.copy()
		for j in range( len(self.trainSet[0]) ):
			self.lvaluesAtts[j] = self.valuesAtts[self.trainSet[0,j]].copy()

		self.NBC = nb.naiveBayes(self.smooth, self.usePrior, meta=self.lvaluesAtts)
		self.NBC.fit( self.trainSet[1:] , self.cl )

		self.orderAtts = self.trainSet[0].copy()

		self.trainSet = []	# it is deleted

		self.isfit = True


	#This function is bad in terms of memory consuption, because in each iteration, it copies the training set
	#	Maybe the best option is to use the trainig set as a Global

	# same parameters than fit
	# prevScore, is the score to be improved, 
	# depth, maximum number of times that this function can be called itself
	def improveStructure(self, prevScore, depth=1):	

		if( depth <= 0 ):
			return

		if( not (1.0 > prevScore >= 0.0)  ):
			raise NameError( "Error: The 'score' to be improved is out of limits" )


		# flag which is useful to recursivity, if there are not changes in the structure, it is not necesary the recursivity anymore
		flag = False
	
		# local copies
		lvaluesAtts = self.valuesAtts.copy()
		ltrainSet = self.trainSet.copy()


		# Begins the process of the Semi Naive Bayes Classifiers
		#  The estructure is modified in order to improve the previous performance
		#	should leave one out cross-validation be used???? ????????????????????????????????????????

		# 80% train, 20# test
		div = int( (len(ltrainSet) - 1 ) * self.validation )
		

		# delete the attributes which have a low MI with the class

		n = len(ltrainSet[0])
		#print("real number of attributes: "+str(n))
		ldel = []		# local list of elements that will be deleted
		ldelNames = []
		for i in range( n ): 		
			xmi = utils.MI( ltrainSet[1:,i], self.cl, self.smooth )
			#print("mi (i,cl): ",xmi)
			if( xmi < self.epsilon ):	# should the same "smooth" be used 
				ldel.append(i)
				ldelNames.append( ltrainSet[0, i] )
		#print("ldel:")
		#print(ldel)

		flagMI = False
		if( len(ldel) > 0 ):
			for i in ldel:
				del lvaluesAtts[ ltrainSet[0,i] ]	# delete from the dictionary
			ltrainSet = np.delete( ltrainSet, ldel, axis=1)			
		else:
			flagMI = True	#if there are not attribute to delete, then it wont be added to self.operations

			#flag = True		#this is not necesary, because the MI will be the same


		n = len(ltrainSet[0])		#update the number of atts
		#print("actual number of attributes: "+str(n))
		#print("actual number of atts in dic: "+str(len( lvaluesAtts )))

		if(n <= 0):
			raise NameError("Error: All the attributtes have been deleted, try to modify the values of epsilon/omega")

		# estimate the CMI, between two attributes given the class
		z = int( ( n*(n - 1) ) / 2.0 )  
		info = np.zeros( ( 3, z ) ).astype(str)
		
		c=0
		for i in range(n):
			for j in range(i+1,n):
				info[0,c] = ltrainSet[0,i]		# instead of the number/index, it is better to use the name of the attribute
				info[1,c] = ltrainSet[0,j]
				info[2,c] = utils.CMI( ltrainSet[1:,i], ltrainSet[1:,j], self.cl, self.smooth )
				c += 1

		info=info[ :, info[2,:].astype(np.float64).argsort()[::-1] ]

		#print("CMI's info:")
		#print(info)
		#print("avg: ", np.average(info[2].astype(float)))
		#print("std: ", np.std(info[2].astype(float)))

		#print("z values: "+str(z))			
		for i in range( z ):	#len(info[0])
			#print("z: "+str(i))
			if( float(info[2,i]) <= self.omega ):		
				break			
			
			# time to compare the 3 diferent options
			# 	eliminate first or second attribute or join the attributes
			lscores = np.zeros(3) 	# 0: del first, 1: del second, 2: join

			flagUpd = False		#  flag that indicates if was updated the structure

			# del attribute 1
			att1 = np.where( ltrainSet[0] == info[0,i] )[0]
			if(len(att1) > 0):

				l2trainSet = np.delete( ltrainSet, att1[0], axis=1)	#train set without att1

				# this dic uses the index as key, not the name of the att
				l2valuesAtts = {}	#lvaluesAtts.copy()
				for j in range( len(l2trainSet[0]) ):
					l2valuesAtts[j] = lvaluesAtts[l2trainSet[0,j]].copy()


				if( len(l2trainSet[0]) != len(l2valuesAtts) ):
					raise NameError("Error (att1) updating the structure")

				nbc = nb.naiveBayes(self.smooth, self.usePrior, meta=l2valuesAtts)
				nbc.fit( l2trainSet[1:(div+1)] , self.cl[:div] )	#, meta=l2valuesAtts )

				pred = nbc.predict(l2trainSet[(div+1):])
				lscores[0] = self.exactMatch( self.cl[div:], pred )
				#print("del att 1: ", lscores[0])

				if( lscores[0] > prevScore):
					prevScore = lscores[0]
					self.NBC = nbc
					self.trainSet = l2trainSet.copy()
					self.valuesAtts = lvaluesAtts.copy()
					del self.valuesAtts[ ltrainSet[0, att1[0] ] ]
			
					if(not flagMI):
						flagMI = True
						self.operations.append( ["del", ldel] )
						self.opeNameAtts.append( ["del", ldelNames ])

					self.operations.append( ["del", att1[0]] )
					self.opeNameAtts.append( ["del", ltrainSet[0, att1[0] ] ] )
					flagUpd = True

				del l2trainSet
			
			
			# del attribute 2
			att2 = np.where( ltrainSet[0] == info[1,i] )[0]
			if(len(att2) > 0):

				l2trainSet = np.delete( ltrainSet, att2[0], axis=1)	#train set without att1

				l2valuesAtts = {}	#lvaluesAtts.copy()
				for j in range( len(l2trainSet[0]) ):
					l2valuesAtts[j] = lvaluesAtts[l2trainSet[0,j]].copy()

				if( len(l2trainSet[0]) != len(l2valuesAtts) ):
					raise NameError("Error (att2) updating the structure")

				nbc = nb.naiveBayes(self.smooth, self.usePrior, meta=l2valuesAtts)
				nbc.fit( l2trainSet[1:(div+1)] , self.cl[:div] )	#, meta=l2valuesAtts )

				pred = nbc.predict(l2trainSet[(div+1):])
				lscores[1] = self.exactMatch( self.cl[div:], pred )
				#print("del att 2: ", lscores[1])

				if( lscores[1] > prevScore):
					if(flagUpd):
						del self.operations[-1]		#deleted the last operation
						del self.opeNameAtts[-1] 

					prevScore = lscores[1]
					self.NBC = nbc
					self.trainSet = l2trainSet.copy()
					self.valuesAtts = lvaluesAtts.copy()
					del self.valuesAtts[ ltrainSet[0, att2[0] ] ]
			
					if(not flagMI):
						flagMI = True
						self.operations.append( ["del", ldel] )
						self.opeNameAtts.append( ["del", ldelNames] )

					self.operations.append( ["del", att2[0]] )
					self.opeNameAtts.append( ["del", ltrainSet[0, att2[0] ] ] )
			
					flagUpd = True

				del l2trainSet
			

			#join the attributtes
			if( (len(att1) > 0) and (len(att2) > 0) ):

				l2trainSet = np.delete( ltrainSet, [att1[0], att2[0] ], axis=1)	#train set without att1 and att2

				# dataMod: contains the data of the attributtes combined, the first element is the combinated name of the attributes
				dataMod = []	
				for j in range( len(ltrainSet) ):
					dataMod.append( ltrainSet[ j, att1[0] ] + ";" + ltrainSet[ j, att2[0] ] )

				# it joins the train set with the new attribute
				l2trainSet = np.column_stack( [l2trainSet, dataMod] )

				l2valuesAtts = {}	#lvaluesAtts.copy()
				for j in range( len(l2trainSet[0]) - 1):
					l2valuesAtts[j] = lvaluesAtts[l2trainSet[0,j]].copy()

				# the las value of the dic is the "combination" of the two attributes
				l2valuesAtts[ len(l2valuesAtts) ] = self.joinList( lvaluesAtts[ ltrainSet[0, att1[0] ] ], lvaluesAtts[ ltrainSet[0, att2[0] ] ] )
				
				if( len(l2trainSet[0]) != len(l2valuesAtts) ):
					raise NameError("Error (join1) updating the structure")

				nbc = nb.naiveBayes(self.smooth, self.usePrior, meta=l2valuesAtts)
				nbc.fit( l2trainSet[1:(div+1)] , self.cl[:div] )	#, meta=l2valuesAtts )

				pred = nbc.predict(l2trainSet[(div+1):])
				lscores[2] = self.exactMatch( self.cl[div:], pred )
				#print("join atts: ", lscores[2])

				if( lscores[2] > prevScore):
					if(flagUpd):
						del self.operations[-1]		#deleted the last operation
						del self.opeNameAtts[-1]

					prevScore = lscores[2]
					self.NBC = nbc
					self.trainSet = l2trainSet.copy()
					self.valuesAtts = lvaluesAtts.copy()
					del self.valuesAtts[ ltrainSet[0, att1[0] ] ]
					del self.valuesAtts[ ltrainSet[0, att2[0] ] ]
					self.valuesAtts[ l2trainSet[0,-1] ] = l2valuesAtts[ len(l2valuesAtts) - 1 ].copy()

			
					if(not flagMI):
						flagMI = True
						self.operations.append( ["del", ldel] )
						self.opeNameAtts.append( ["del", ldelNames] )

					self.operations.append( [ "join", [ att1[0], att2[0] ] ] )
					self.opeNameAtts.append( ["join", [ ltrainSet[0, att1[0] ], ltrainSet[0, att2[0]] ] ] )

					flagUpd = True

				del l2trainSet


			if(flagUpd):	# if the structure was updated, we need to update the local structure
				ltrainSet = self.trainSet.copy()
				lvaluesAtts = self.valuesAtts.copy()
				if( len(ltrainSet[0]) != len(lvaluesAtts) ):
					raise NameError("Error updating the structure")
				#print("iterarion ok: "+str(i))
				#print(lvaluesAtts)

			
			#print("Scores: del att1, del att2, join atts")
			#print(lscores)
			#exit()
			#if( ( len(np.where( ltrainSet[0] == info[0,i] )[0]) > 0 ) and ( len(np.where( ltrainSet[0] == info[1,i] )[0]) > 0 ) ):


	# it joins two list 
	# l1=["1","2","3"]
	# l2=["a","b"]
	# joinList(l1,l2) wil return ["1-a","1-b","2-a","2-b","3-a","3-b"]
	def joinList(self,listA, listB):
		newl = []

		for x in listA:
			for y in listB:
				newl.append( x + ";" + y )
	
		return np.array(newl)


	# It applies the operations to a new set (for example, to the test set)
	def applyOperations(self, data):
		self.checkIfFit()	# first check if the classifier is already trained
		daux = data.copy()
		#print("daux main: ")
		#print(daux)
		for x in self.operations:
			if(x[0] == "del"):
				daux = np.delete( daux, x[1], axis=1)
			elif( x[0] == "join"):
				# first to combine the attributes
				dataMod = []	
				for j in range( len(daux) ):
					dataMod.append( daux[ j, x[1][0] ] + ";" + daux[ j, x[1][1] ] )
				
				# then delete the attributes
				daux = np.delete( daux, x[1], axis=1)	#train set without att1 and att2

				#finally concatenate the new attribute
				daux = np.column_stack( [daux, dataMod] )
			else:
				raise NameError("Error: unknown operation ")
			#print("daux: ")
			#print(daux)

		return daux


	#returns de probabilities of each class
	def predict_proba(self,testSet):
		self.checkIfFit()	# first check if the classifier is already trained
		return self.NBC.predict_proba( self.applyOperations(testSet).astype(str))

	def predict_log_proba(self,testSet):
		self.checkIfFit()	# first check if the classifier is already trained
		return self.NBC.predict_log_proba( self.applyOperations(testSet).astype(str))

	def predict(self,testSet):
		self.checkIfFit()	# first check if the classifier is already trained
		return self.NBC.predict( self.applyOperations(testSet).astype(str))

	def exactMatch(self, real, prediction):
		if(len(real.astype(str)) != len(prediction)):
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
		
