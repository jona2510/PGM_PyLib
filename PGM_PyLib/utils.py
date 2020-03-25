"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

#libraries
import numpy as np

# Mutual Information
# I (X;Y)
def MI(X,Y,smooth=0.1):
	if( len(X) != len(Y) ):
		raise NameError("The size of the vectors are different")

	#print("smooth: "+str(smooth))
	if( smooth < 0.0):
		raise NameError("Smooth has to be >= 0")

	valuesX = list(set(X))
	valuesY = list(set(Y))	

	PX = np.zeros( len(valuesX) )		# P(X)	
	PY = np.zeros( len(valuesY) )		# P(Y)	
	PXY = np.zeros( ( len(valuesX), len(valuesY) ) )		# P(X,Y) 	
	
	# counting from data

	for i in range(len(valuesX)):
		PX[i]= np.sum( X == valuesX[i] )
	
	for i in range(len(valuesY)):
		PY[i]= np.sum( Y == valuesY[i] )

	for i in range(len(valuesX)):
		x = set( np.where(X == valuesX[i])[0] )

		for k in range(len(valuesY)):
			#these values could be stored instead of being recalculated in each iteration of the outer loop
			y = set( np.where(Y == valuesY[k])[0] )
			PXY[i,k] = len( x & y )	

	if( (np.sum(PY) != np.sum(PX)) or (np.sum(PX) != np.sum(PXY))  ):
		raise NameError("There was an error estimating the probabilities")

	#smooth
	PX += smooth
	PY += smooth
	PXY += smooth

	#joint probabilities
	PX /= np.sum(PX)
	PY /= np.sum(PY)
	PXY /= np.sum(PXY)

	I = 0.0
	for i in range(len(valuesX)):
		for k in range(len(valuesY)):
			I += PXY[i,k] * np.log(  PXY[i,k] / ( PX[i] * PY[k] ) )

	if(I < 0):
		return 0
	return I

# Conditional Mutual Information
# I (X;Y | Z)
def CMI(X,Y,Z,smooth=0.1):

	if( (len(X) != len(Y)) or (len(Y) != len(Z)) ):
		raise NameError("The size of the vectors are different")

	if( smooth < 0.0):
		raise NameError("Smooth has to be >= 0")

	valuesX = list(set(X))
	valuesY = list(set(Y))
	valuesZ = list(set(Z))

	#print("valuesX"+str(len(valuesX)))
	#print("valuesY"+str(len(valuesY)))
	#print("valuesZ"+str(len(valuesZ)))

	PZ = np.zeros( len(valuesZ) )		# P(Z)	
	PXZ = np.zeros( ( len(valuesX), len(valuesZ) ) )		# P(X,Z) 
	PYZ = np.zeros( ( len(valuesY), len(valuesZ) ) )		# P(Y,Z)
	PXYZ = np.zeros( ( len(valuesX), len(valuesY), len(valuesZ) ) )		# P(X,Y,Z)

	# counting from data

	for i in range(len(valuesZ)):
		PZ[i]= np.sum( Z == valuesZ[i] )
	 
	for i in range(len(valuesX)):
		x = set( np.where(X == valuesX[i])[0] )

		for k in range(len(valuesY)):
			y = set( np.where(Y == valuesY[k])[0] )

			for j in range(len(valuesZ)):
				z = set( np.where(Z == valuesZ[j])[0] )

				PXYZ[i,k,j] = len( (x & y) & z )	

				if(k==0): 
					#print("---P(X,Z)--- se ejecutara len(valuesX) veces")
					PXZ[i,j] = len( x & z )

				if(i==0):	#
					#print("---P(Y,Z)---")
					PYZ[k,j] = len( y & z )
	"""
	print("PZ:")
	print(PZ)
	print("PXZ:")
	print(PXZ)
	print("PYZ:")
	print(PYZ)
	print("PXYZ:")
	print(PXYZ)
	"""	


	if( (np.sum(PZ) != np.sum(PXZ)) or (np.sum(PXZ) != np.sum(PYZ)) or (np.sum(PYZ) != np.sum(PXYZ)) ):
		raise NameError("There was an error estimating the probabilities")

	#smooth
	PZ += smooth
	PXZ += smooth
	PYZ += smooth
	PXYZ += smooth

	#joint probabilities
	PZ /= np.sum(PZ)	
	PXZ /= np.sum(PXZ)
	PYZ /= np.sum(PYZ)
	PXYZ /= np.sum(PXYZ)

	#print("PXYZ***************")
	#print(PXYZ)

	I=0.0
	for i in range(len(valuesX)):
		for k in range(len(valuesY)):
			for j in range(len(valuesZ)):
				I += PXYZ[i,k,j]*np.log( ( PZ[j] * PXYZ[i,k,j] )/( PXZ[i,j] * PYZ[k,j] ) )

	if(I < 0):
		return 0
	return I



