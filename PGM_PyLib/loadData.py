"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

from scipy.io import arff
import numpy as np
import pandas as pd

"""
Check if all the attributes are nominal, if someone isn't it then raise a exception
"""
def areAllNominals(meta):
	for x in meta:
		if(meta[x][0] != "nominal"):
			#return False
			raise NameError("All the attributes have to be nominals")

	#return True

"""
Split the data in data and classes
THE CLASSES HAS TO BE AT THE END OF DATA!!!!
"""
def splitDataClass(data,numClasses=1):
	if(numClasses<1 or numClases >= len(data[0])):
		raise NameError("The number of classes should be: 1<= value < (number columns of data) ")		




"""
this returns the data and 'meta'data 
But first check that all the attrbuttes are nominals
"""
def loadArffFile(nameArff):
	data,meta = arff.loadarff(nameArff)

	areAllNominals(meta)
	#transform to numpy
	data=(pd.DataFrame(data) ).to_numpy(str)

	return data,meta

"""
this returns the data and 'meta'data
"""
def loadArffFull(nameArff):
	data,meta = arff.loadarff(nameArff)

	#areAllNominals(meta)
	#transform to numpy
	data=(pd.DataFrame(data) ).to_numpy(str)

	return data,meta


"""
this returns the data splitted in train and test sets, also the 'meta'data 
"""
def loadArffSplit(nameArff,trainPercentage):

	#if(trainPercentage<=0 or trainPercentage>1):
	if(not (0 < trainPercentage <= 1) ):
		raise NameError("The percentage for training set has to be: 0< value <=1 ")

	data,meta = arff.loadarff(nameArff)
	areAllNominals(meta)

	data=(pd.DataFrame(data) ).to_numpy(str)

	d = int(np.ceil( len(data)*trainPercentage ))	#

	#si no se copian directamente los datos de la matriz data[0], solo se envia un apuntador que tiene acceso a la fraccion correspondiente de la matriz de data[0]


	train =data[:d]	#train
	test  =data[d:]	#test

	return train,test,meta

