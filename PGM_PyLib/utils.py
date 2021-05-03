"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

#libraries
import numpy as np

from scipy.stats import chi2 as c2
from scipy.stats import chi2_contingency as c2c

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
	
	#print("PZ:")
	#print(PZ)
	#print("PXZ:")
	#print(PXZ)
	#print("PYZ:")
	#print(PYZ)
	#print("PXYZ:")
	#print(PXYZ)
		


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


def CMI_setZ(X,Y,Z,smooth=0.1):
	def getZpos(lenZs, lpos):
		"""
		return the position for the array z, that is, list to integer

		nvaluesZ contains the number of values for each variable in z
		lpos, a list with the position required

		"""
		ma = 1
		#pos = lpos[0]
		pos = lpos[-1]
		#for i in range(1,len(lenZs)):			
		for i in range( len(lenZs) -2 ,-1,-1):
			#ma *= lenZs[i-1]
			ma *= lenZs[i+1]
			pos += ma*lpos[i]
		return pos

	def pos2list(pos, lenZs):
		"""
		returns the list of the position with respect to pos, that is, integer to list
		pos, integer, position
		nvaluesZ contains the number of values for each variable in z
		"""
		l = [0 for i in range( len(lenZs) )]
		m = [0 for i in range( len(lenZs) )]
		m[-1] = 1
		for i in range( len(lenZs) -2 ,-1,-1):
			m[i] = m[i+1] * lenZs[i+1]
		pr = pos
		for i in range(len(m)):
			l[i] = pr // m[i]
			pr = pr % m[i]
		if(pr != 0):
			raise NameError("failed to create convert position to list!")
		return l

	def setZ_rec(valuesZ, lpos, pvmz, pos, setData):
		"""
		valuesZ, a list with contains in each position the values that can take the component of z
		lpos, vector with full position
		pos, position to evaluate lpos
		pvmz, list which contains in each position the position (in data) where the respective value occurs
		setData: set with the positon of the accomulative data
		"""
		actualSet = setData & pvmz[pos][ lpos[pos] ]
		if( (len(actualSet) == 0) or ( (pos+1) >= len(lpos) ) ):
			#finish when the set is empty or when finish of evaluating lpos	
			return actualSet	

		return setZ_rec(valuesZ, lpos, pvmz, pos+1, actualSet )


	def setZ(valuesZ, lpos, pvmz):	
		"""
		return the set of position of the data which have the combination in lpos

		valuesZ, a list with contains in each position the values that can take the component of z
		lpos, vector with full position
		pvmz, list which contains in each position the position (in data) where the respective value occurs
		"""
		setD = pvmz[0][ lpos]
		#for 
		return countZ_rec(valuesZ, lpos, pvmz, 1, pvmz[0][ lpos[0] ] )

		#for i in range(len(valuesZ)):
		#	PZ[i] = np.sum( Z == valuesZ[i] )
			
	
	shz = np.shape(Z)
	
	if(len(shz) != 2):
		raise NameError("Z has to be a matrix, a ndarray of shape (n_data,2)")

	if( (len(X) != len(Y)) or (len(Y) != shz[0]) ):
		raise NameError("The size of the vectors are different")

	if( smooth < 0.0):
		raise NameError("Smooth has to be >= 0")

	valuesX = list(set(X))
	valuesY = list(set(Y))
	valuesZ = [ list( set(Z[:,i]) ) for i in range( shz[1] ) ]	#list(set(Z))

	lenZs = [ len(valuesZ[i]) for i in range(shz[1]) ]
	nvaluesZ = 1
	for i in range(shz[1]):
		nvaluesZ *= lenZs[i]
		
	#print("valuesX"+str(len(valuesX)))
	#print("valuesY"+str(len(valuesY)))
	#print("valuesZ"+str(len(valuesZ)))

	PZ = np.zeros( nvaluesZ )		# P(Z)	
	PXZ = np.zeros( ( len(valuesX), nvaluesZ ) )		# P(X,Z) 
	PYZ = np.zeros( ( len(valuesY), nvaluesZ ) )		# P(Y,Z)
	PXYZ = np.zeros( ( len(valuesX), len(valuesY), nvaluesZ ) )		# P(X,Y,Z)


	#obtan the postion where the data occurs for each value	
	# this requieres memory

	pvmx = []
	pvmy = []
	pvmz = [ [] for i in range(shz[1]) ]

	for x in range(shz[1]):
		for i in valuesZ[x]:
			pvmz[x].append( set(np.where(Z[:,x] == i)[0]) )

	for i in valuesX:
		pvmx.append( set(np.where(X == i)[0]) )

	for i in valuesY:
		pvmy.append( set(np.where(Y == i)[0]) )

	#print("pvmz")
	#print(pvmz)


	# counting from data

	#for i in range(len(valuesZ)):
	for i in range(nvaluesZ):
		posl = pos2list(i,lenZs)
		setl = pvmz[0][posl[0]]
		for j in range(1,len(posl)):
			setl = setl & pvmz[j][posl[j]]
			if(len(setl)==0):
				break
		PZ[i] = len(setl)
		#PZ[i]= np.sum( Z == valuesZ[i] )
	 
	for i in range(len(valuesX)):
		#x = set( np.where(X == valuesX[i])[0] )
		x = pvmx[i]	#set( np.where(X == valuesX[i])[0] )

		for k in range(len(valuesY)):
			#y = set( np.where(Y == valuesY[k])[0] )
			y = pvmy[k]		#set( np.where(Y == valuesY[k])[0] )

			#for j in range(len(valuesZ)):
			for j in range(nvaluesZ):
				#z = set( np.where(Z == valuesZ[j])[0] )
				posl = pos2list(j,lenZs)
				z = pvmz[0][posl[0]]
				for l in range(1,len(posl)):
					z = z & pvmz[l][posl[l]]
					if(len(z)==0):
						break

				PXYZ[i,k,j] = len( (x & y) & z )	

				if(k==0): 
					#print("---P(X,Z)--- se ejecutara len(valuesX) veces")
					PXZ[i,j] = len( x & z )

				if(i==0):	#
					#print("---P(Y,Z)---")
					PYZ[k,j] = len( y & z )
	
	#print("PZ:")
	#print(PZ)
	#print("PXZ:")
	#print(PXZ)
	#print("PYZ:")
	#print(PYZ)
	#print("PXYZ:")
	#print(PXYZ)
		


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
			#for j in range(len(valuesZ)):
			for j in range(nvaluesZ):
				I += PXYZ[i,k,j]*np.log( ( PZ[j] * PXYZ[i,k,j] )/( PXZ[i,j] * PYZ[k,j] ) )

	if(I < 0):
		return 0
	return I



