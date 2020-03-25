"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

# all this metrics are for Hierarchical Classification
#	that is, the labels/classes are arranged in a predefined structure
#	and the instances are associated to a subset of the labels while complain the hierarchical constraint
#
# matrix = rows x coloumns
# matrix = instances x labels



#libraries
import numpy as np


def check_error(shapeR, shapeP):
	if( (len(shapeR) != 2) or (len(shapeP) != 2) ):
		raise NameError( "Error: you has to provide two two-dimensional numpy matrices!" )

	if( (shapeR[0] != shapeP[0]) or (shapeR[1] != shapeP[1]) ):
		raise NameError( "Error: The dimensions of the matrices are different!" )		



# Exact match
def exactMatch(real, prediction):
	shR = np.shape(real)
	shP = np.shape(prediction)

	check_error(shR, shP)		

	c=0.0
	for i in range(shR[0]):
		if( np.all(real[i] == prediction[i]) ):		# np.all returns true if all the elements are true
			c += 1

	return (c/shR[0])



# Accuracy
def accuracy(real, prediction):
	shR = np.shape(real)
	shP = np.shape(prediction)

	check_error(shR, shP)		

	acc = 0.0		

	for i in range(shR[0]):			# walks over the intances
		union = 0.0				# sum of predicted and real
		intersection = 0.0		# correctly predicted

		for j in range(shR[1]):		# walks over the labels
			#if(real[i,j] == prediction[i,j]):
			if(real[i,j] == 1):
				if(prediction[i,j] == 1):
					intersection += 1
				union += 1
			else:
				if(prediction[i,j] == 1):
					union += 1						
	
		acc += intersection / union

	return ( acc / shR[0] )



# Hamming Loss
#	it is used in Hamming accuracy
def hammingLoss(real, prediction):
	shR = np.shape(real)
	shP = np.shape(prediction)

	check_error(shR, shP)	

	acc = 0.0		

	for i in range(shR[0]):			# walks over the intances
		for j in range(shR[1]):		# walks over the labels
			if(real[i,j] != prediction[i,j]):	#symmetric difference
				acc += 1

	return (acc / (shR[0] * shR[1]) )


# Hamming accuracy
def hammingAccuracy(real, prediction):

	hLoss = hammingLoss(real, prediction)

	return (1 - hLoss)


# hierarchical recall hR
#
def hRecall(real, prediction, check=True):	

	shR = np.shape(real)
	shP = np.shape(prediction)

	if(check):
		check_error(shR, shP)

	nReal = 0.0				# number of "real" labels
	intersection = 0.0		# correctly predicted

	for i in range(shR[0]):			# walks over the intances
		for j in range(shR[1]):		# walks over the labels
			if(real[i,j] == 1):
				if(prediction[i,j] == 1):
					intersection += 1
				nReal += 1

	if(nReal == 0):
		print("WARNING: the number of real associations in the whole dataset was zero")
		return 0
	return ( intersection / nReal )


# hierarchical precision hP
#
def hPrecision(real, prediction, check=True):

	shR = np.shape(real)
	shP = np.shape(prediction)

	if(check):
		check_error(shR, shP)

	nPred = 0.0				# number of "predicted" labels
	intersection = 0.0		# correctly predicted

	for i in range(shR[0]):			# walks over the intances
		for j in range(shR[1]):		# walks over the labels
			if(prediction[i,j] == 1):
				if(real[i,j] == 1):
					intersection += 1
				nPred += 1

	if(nPred == 0.0):
		print("WARNING: the number of predictions in the whole dataset was zero")
		return 0
	else:
		return ( intersection / nPred )

# hierarchical F measure (hF)
def hFmeasure(real, prediction, check=True):

	hP = hPrecision(real, prediction, True)
	hR = hRecall(real, prediction, False)

	if(hP == hR == 0):
		print("WARNING: hierarchical Precision and hierarchical Recall were zero")
		return 0
	else:
		return ( (2 * hP * hR) / (hP + hR) )


