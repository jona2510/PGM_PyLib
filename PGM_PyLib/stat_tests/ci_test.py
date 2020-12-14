"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np
from scipy.stats import chi2 as c2
from scipy.stats import chi2_contingency as c2c


class chi2_ci_test:

	def __init__(self, significance=0.05, correction=False, lambda_=None,smooth=0.0):
		self.significance = significance
		self.correction = correction
		self.lambda_= lambda_
		self.smooth = smooth


	def test(self,X,Y,Z):
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
			raise NameError("Z has to be a matrix, a ndarray of shape (n_samples ,m_variables)")

		if( (len(X) != len(Y)) or (len(Y) != shz[0]) ):
			#print("X: ",len(X),", Y: ",len(Y),", Z: ",shz[0])
			raise NameError("The size of the vectors are different")
		

		if( self.smooth < 0.0):
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

		# for chi2, table
		ctable = np.zeros(( len(valuesX), len(valuesY) )) 	#frecuency table (contingency table)

		#obtan the position where the data occurs for each value	
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

		#print("pvmx:\n",pvmx)
		#print("pvmy:\n",pvmy)

		# the chi2 value for test
		c2s = c2.ppf(self.significance, (len(valuesX) - 1) * (len(valuesY) - 1) )

		# fill ctable with frequencies
		c2values = np.zeros(nvaluesZ)
		for i in range(nvaluesZ):
			posl = pos2list(i,lenZs)
			setl = pvmz[0][posl[0]]
			for j in range(1,len(posl)):
				setl = setl & pvmz[j][posl[j]]
				if(len(setl)==0):
					break

			#print("setz: ",setl)

			ctable.fill(0)
			if(len(setl)>0):
				for x in range(len(valuesX)): # walk over X
					setlx = setl & pvmx[x]
					#print("setlx: ",setlx)
					for y in range(len(valuesY)):
						ctable[x,y] = len( setlx & pvmy[y] )
						#print("ctable[",x,",",y,"] = ",ctable[x,y])
		
				if( np.any( ctable < 5 ) ):
					print("Warning!!, in contingency table of z-position ",posl," there is/are cell with less than 5 elements!")

			else:
				c2values.fill(0)
				print("Warning!!, contingency table of z-position ",posl," was filled with zeros!!")

			#apply smooth
			ctable = ctable + self.smooth
			#print("ctable:\n",ctable)
			c2values[i],p,dof,expected = c2c(ctable,correction=self.correction,lambda_=self.lambda_)

			if(c2values[i] < c2s):	# if the chi2 test is not fullfilled,
				return False 

	
		#print("c2values (chi2 test applied to each combination of z)")
		#print(c2values)

		c2s = c2.ppf(1-self.significance, nvaluesZ )
		# apply chi2 test to c2values
		va = c2.cdf( sum(c2values) , nvaluesZ )
		#print(".-. ",va)
		pvalue = 1 - va
		#print("c2s: ",c2s)
		#print("sum chi2s: ",sum(c2values))
		#print("pv: ",pvalue)
		
		if(sum(c2values) < c2s):
			return True
		else:
			return False

		#print("pi value X,Y | Z:")
		#print(1-pvalue)
		#print()

		#return pvalue


