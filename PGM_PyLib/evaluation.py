"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

# all this metrics are for multiclass classification 
#	that is, there are two or more classes,
#	but the instances are asociated to only one class


#libraries
import numpy as np


def check_error(shapeR, shpaeP):
	if( (len(shapeR) != 1) or (len(shapeP) != 1) ):
		raise NameError( "Error: you has to provide two vectors!" )

	#if(shapeR[0] != shapeP[0]):
	if( (shapeR[0] != shapeP[0]) or (shapeR[1] != shapeP[1]) ):
		raise NameError( "Error: The dimension of the vectors is different!" )		


#confusion matrix
def confusionMatrix(real, prediction):
	shR = np.shape(real)
	shP = np.shape(prediction)

	check_error(shR, shP)

	classes = list(set(real))
	nc = len(classes)

	dp = {}
	for i in range(nc):
		dp[classes[i]] = i

	cm = np.zeros( (nc, nc) )
	
	for i in range(shR[0]):
		cm[ dp[ real[i] ], dp[ prediction[i] ] ] += 1


	return (classes, cm)


# Exact match
def exact_match(real, prediction):
	shR = np.shape(real)
	shP = np.shape(prediction)

	check_error(shR, shP)		

	c=0.0
	for i in range(shR[0]):
		#if(real[i] == prediction[i]):
		if( np.all(real[i] == prediction[i]) ):		# np.all returns true if all the elements are true
			c += 1

	return (c/shR[0])

# sensitivity, recall, hit rate, or true positive rate (TPR)
#
#	TPR = TP/P = TP / (TP + FN)
def recallFEC(real, prediction, check=True):	# recall For Each Class

	shR = np.shape(real)
	shP = np.shape(prediction)

	if(check):
		check_error(shR, shP)

	classes = list(set(real))
	nc = len( classes )		#number of classes
	sc = np.zeros( nc )		#score of each class

	for i in range(nc):
		TP = 0.0	# True positive
		TN = 0.0	# True negative
		FP = 0.0	# False positive
		FN = 0.0	# False negative

		for j in range(shR):
			if( classes[i] == real[j] ):	
				if( real[j] == prediction[j] ):
					TP += 1
				else:
					FN += 1
			else:
				if( classes[i] != prediction[j] ):
					TN += 1
				else:
					FP += 1

		sc[i] = TP / (TP + FN)

	return (classes, sc)

def recall(real, prediction, check=True):	# recall
	
	a = recallFEC(real, prediction, check)

	return np.average( a[1] )

#precision or positive predictive value (PPV)
#
# PPV = TP / (TP + FP)
def precisionFEC(real, prediction, check=True):		# precision For Each Class

	shR = np.shape(real)
	shP = np.shape(prediction)

	if(check):
		check_error(shR, shP)

	classes = list(set(real))
	nc = len( classes )		#number of classes
	sc = np.zeros( nc )		#score of each class

	for i in range(nc):
		TP = 0.0	# True positive
		TN = 0.0	# True negative
		FP = 0.0	# False positive
		FN = 0.0	# False negative

		for j in range(shR):
			if( classes[i] == real[j] ):	
				if( real[j] == prediction[j] ):
					TP += 1
				else:
					FN += 1
			else:
				if( classes[i] != prediction[j] ):
					TN += 1
				else:
					FP += 1

		sc[i] = TP / (TP + FP)

	return (classes, sc)

def precision(real, prediction, check=True):		

	a = precisionFEC(real, prediction, check)

	return np.average(a[1])


# f1 score
def f1FEC(real, prediction, check=True):	# f1 score For Each Class

	shR = np.shape(real)
	shP = np.shape(prediction)

	if(check):
		check_error(shR, shP)

	tpr = recallFEC(real, prediction, False)
	ppv = precisionFEC(real, prediction,False)

	if(tpr[0] != ppv[0]):	
		# the order of the classes id different
		raise NameError("New implmentation of 'f1 score' is required")


	nc = len( len(trp[1]) )		#number of classes
	sc = np.zeros( nc )		#score of each class

	for i in range( nc ):
		sc[i] = (2 * tpr[1][i] * ppv[1][i]) / (tpr[1][i] + ppv[1][i])
		

	return (tpr[0], sc)
	#return ( (2 * tpr * ppv) / (tpr +  ppv) )
	
def f1(real, prediction, check=True):

	a = f1FEC(real, prediction, check)

	return np.average(a[1])


