"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np 


def smoothing(rmrf, row, col):
	"""
	similar values are preferred
	First order MRF 
	
	rmrf:	the matrix wich conntains the data of the RMRF
	row,col:	position of the variable of interest
	"""
	sh = np.shape(rmrf)	

	z = 0


	"""
	ulc------ul-------urc
	|-------------------|
	|-------------------|
	ll-----------------rl
	|-------------------|
	|-------------------|
	blc------bl-------brc
	"""
	#print(row,col)
	if(row == 0):
		if(col == 0):	# upper-left corner (upc)
			z = abs(rmrf[0,0] - rmrf[0,1]) + abs(rmrf[0,0] - rmrf[1,0])
		elif(col == (sh[1]-1)): # upper-right corner (urc)
			z = abs(rmrf[0,-1] - rmrf[0,-2]) + abs(rmrf[0,-1] - rmrf[1,-1])
		else: # upper line (ul)
			z = abs(rmrf[row,col] - rmrf[row,col-1]) + abs(rmrf[row,col] - rmrf[row,col+1]) + abs(rmrf[row,col] - rmrf[row+1,col])

	elif(row == (sh[0]-1)):
		if(col == 0):	# bottom-left corner (blc)
			z = abs(rmrf[row,col] - rmrf[row,col+1]) + abs(rmrf[row,col] - rmrf[row-1,col])
		elif(col == (sh[1]-1)): # botton-right corner (brc)
			z = abs(rmrf[row,col] - rmrf[row,col-1]) + abs(rmrf[row,col] - rmrf[row-1,col])
		else: # bottom line (bl)
			z = abs(rmrf[row,col] - rmrf[row,col-1]) + abs(rmrf[row,col] - rmrf[row,col+1]) + abs(rmrf[row,col] - rmrf[row-1,col])
	else:
		if(col == 0):	# left line (ll)
			z = abs(rmrf[row,col] - rmrf[row+1,col]) + abs(rmrf[row,col] - rmrf[row-1,col]) + abs(rmrf[row,col] - rmrf[row,col+1])
		elif(col == (sh[1]-1)):	# right line (rl)
			z = abs(rmrf[row,col] - rmrf[row+1,col]) + abs(rmrf[row,col] - rmrf[row-1,col]) + abs(rmrf[row,col] - rmrf[row,col-1])
		else:	# rest
			z = abs(rmrf[row,col] - rmrf[row+1,col]) + abs(rmrf[row,col] - rmrf[row-1,col]) + abs(rmrf[row,col] - rmrf[row,col+1]) + abs(rmrf[row,col] - rmrf[row,col-1])

	return z


class RMRF:
	"""
	Implementation of Regular Markov Random Fields (RMRF)
	"""

	#def __init__(self, states=None, observations=None, prior=None, transition=None, observation=None ):
	#def __init__(self, states, rmrf, order=1, potencials = None ):
	def __init__(self, states, rmrf, order=1, potencials = None ):
		"""
		Constructor of (R)egular (M)arkov (R)andom (F)ields

		states 	: 		set of states
		rmrf 	:		numpy matrix of shape (x,y) with initial values
		order	:		useless
		potencials	:		useless
		"""

		try:
			spl = states.split("-")
			self.range = True
		except:
			self.range = False			

		if(self.range):
			self.smin = int(spl[0])
			self.smax = int(spl[1])
			self.nstates = self.smax - self.smin
			if( self.nstates < 2):
				raise NameError("Error: The range has to contain at least two states")
		else:
			self.nstates = len(states)
			if(self.nstates < 2):
				raise NameError("Error: The set of states has to contain at least two states.")
				
		self.states = states	# set of states or range

		self.shrmrf = np.shape(rmrf)
		if(len(self.shrmrf) != 2):
			raise NameError("Error: rmrf has to be a numpy matrix of shape (x,y)")
		self.rmrf = rmrf.copy()
		self.potencials = potencials



	def ICM(self, Uf=smoothing, threshold=0.01, maxIterations=10, variant="MAP"):
		raux = self.rmrf.copy()

		for it in range (maxIterations):
			for i in range(self.shrmrf[0]):		#rows
				for j in range(self.shrmrf[1]):	#cols
					t =  self.getDifferentValue(self.rmrf[i,j])	#alternative value
					ufi = Uf(self.rmrf,i,j)	#  with position, son only evaluate the local point of interest
					aux = self.rmrf[i,j]
					self.rmrf[i,j] = t				
					uft = Uf(self.rmrf,i,j)	#  with position, so only evaluate the local point of interest
					if(uft > ufi):
						self.rmrf[i,j] = aux # return original value

			if(np.all(raux == self.rmrf)):	#if there are no change, then finish
				print("Succesfully finish") 
				break
			else:
				raux = self.rmrf.copy()
		return self.rmrf


	def metropolis(self, Uf=smoothing, threshold=0.01, maxIterations=10, prob=0.4, variant="MAP"):
		raux = self.rmrf.copy()

		for it in range (maxIterations):
			for i in range(self.shrmrf[0]):		#rows
				for j in range(self.shrmrf[1]):	#cols
					t =  self.getDifferentValue(self.rmrf[i,j])	#alternative value
					ufi = Uf(self.rmrf,i,j)	#  with position, son only evaluate the local point of interest
					aux = self.rmrf[i,j]
					self.rmrf[i,j] = t				
					uft = Uf(self.rmrf,i,j)	#  with position, so only evaluate the local point of interest
					if(uft > ufi):
						if(np.random.rand() > prob):
							self.rmrf[i,j] = aux # return original value

			if(np.all(raux == self.rmrf)):	#if there are no change, then finish
				print("Succesfully finish") 
				break
			else:
				raux = self.rmrf.copy()
		return self.rmrf


	def getDifferentValue(self,value):
		while True:
			if(self.range):			
				nv = np.random.randint(self.smin,self.smax)	#a random value
			else:
				nv = self.states[ np.random.randint(self.nstates) ]	#a random value
			if(nv != value):
				return nv

	def sAnnealing(self, Uf=smoothing, threshold=0.01, maxIterations=10, Temp=0.9, tempReduction=0.7, variant="MAP"):
		raux = self.rmrf.copy()

		for it in range (maxIterations):
			for i in range(self.shrmrf[0]):		#rows
				for j in range(self.shrmrf[1]):	#cols
					t =  self.getDifferentValue(self.rmrf[i,j])	#alternative value
					ufi = Uf(self.rmrf,i,j)	#  with position, son only evaluate the local point of interest
					aux = self.rmrf[i,j]
					self.rmrf[i,j] = t				
					uft = Uf(self.rmrf,i,j)	#  with position, son only evaluate the local point of interest
					if(uft > ufi):
						d = uft - ufi
						pt = np.power(np.e,-d/Temp)	# probability of keeping the value with higher energy
						if(np.random.rand() > pt):
							self.rmrf[i,j] = aux # return original value

			if(np.all(raux == self.rmrf)):	#if there are no change, then finish
				print("Succesfully finish") 
				break
			else:
				raux = self.rmrf.copy()
		return self.rmrf

	def functionMPM(self,previous,current):
		"""
		This fuction adds one to each state (in current) with respect to the new values of the MRF
		previous: is a numpy matrix with dictionaries as elements.
		current: is a numpy matrix with current values of the MRF
		"""
		for i in range(self.shrmrf[0]):		#rows
			for j in range(self.shrmrf[1]):	#cols
				if(current[i,j] in previous[i,j]):
					previous[i,j][ current[i,j] ] += 1
				else:
					previous[i,j][ current[i,j] ] = 1

		return previous

	def returnMPM(self,mpm):
		"""
		return the matrix with the correct values for the MPM procedure
		mpm: is a numpy matrix with dictionaries as elements.
		"""
		res = np.zeros(self.shrmrf,dtype=int) - 1
		for i in range(self.shrmrf[0]):		#rows
			for j in range(self.shrmrf[1]):	#cols
				for k in mpm[i,j].keys():
					if( mpm[i,j][k] > res[i,j] ):
						res[i,j] = k

		return res


	def inference(self, Uf=smoothing, maxIterations=10, Temp=1.0, tempReduction=1.0, optimal="MAP"):
		"""
		General method, 
		Temp == 1, and tempReduction == 1, ICM
		Temp != 1, and tempReduction == 1, metropolis
		Temp != 1, and tempReduction != 1, simulated anealing
		"""
		raux = self.rmrf.copy()

		if(optimal == "MPM"):
			#create the variable which contains the number that each state is used.
			mpm = np.array( [[{} for i in range(self.shrmrf[1])] for j in range(self.shrmrf[0]) ] )
			mpm = self.functionMPM(mpm, raux)
		elif(optimal != "MAP"):
			raise NameError("Error, optimal only can take the values 'MPM' or 'MAP'") 
			

		for it in range (maxIterations):
			for i in range(self.shrmrf[0]):		#rows
				for j in range(self.shrmrf[1]):	#cols
					t =  self.getDifferentValue(self.rmrf[i,j])	#alternative value
					ufi = Uf(self.rmrf,i,j)	#  with position, son only evaluate the local point of interest
					aux = self.rmrf[i,j]
					self.rmrf[i,j] = t				
					uft = Uf(self.rmrf,i,j)	#  with position, son only evaluate the local point of interest
					if(uft >= ufi):
						if(tempReduction == 1):
							if(Temp == 1):	# ICM
								self.rmrf[i,j] = aux # return original value								
							else:			# Metropilis
								if(np.random.rand() > Temp):
									self.rmrf[i,j] = aux # return original value
						else:				#simulated annealing
							d = uft - ufi						
							pt = np.power(np.e,-d/Temp)	# probability of keeping the value with higher energy
							if(np.random.rand() > pt):
								self.rmrf[i,j] = aux # return original value

			if(np.all(raux == self.rmrf)):	#if there are no change, then finish
				print("Succesfully finish, iteration: "+str(it)) 
				break
			else:
				raux = self.rmrf.copy()
			# the last iteration is not counted
			if(optimal == "MPM"):
				mpm = self.functionMPM(mpm, raux)

			Temp *= tempReduction

		if(optimal == "MPM"):
			return self.returnMPM(mpm)
		else:
			return self.rmrf



#******************************************************************************************+
#******************************************************************************************+
#******************************************************************************************+
#******************************************************************************************+

def smoothingImages(rmrf, observation, row, col):
	"""
	similar values are preferred
	First order MRF 
	
	rmrf:	the matrix wich conntains the data of the RMRF
	row,col:	position of the variable of interest
	"""
	lamda = 4

	sh = np.shape(rmrf)	
	z = 0


	"""
	ulc------ul-------urc
	|-------------------|
	|-------------------|
	ll-----------------rl
	|-------------------|
	|-------------------|
	blc------bl-------brc
	"""
	#print(row,col)
	if(row == 0):
		if(col == 0):	# upper-left corner (upc)
			z = abs(rmrf[0,0] - rmrf[0,1]) + abs(rmrf[0,0] - rmrf[1,0]) + lamda*abs(rmrf[row,col] - observation[row,col])
		elif(col == (sh[1]-1)): # upper-right corner (urc)
			z = abs(rmrf[0,-1] - rmrf[0,-2]) + abs(rmrf[0,-1] - rmrf[1,-1]) + lamda*abs(rmrf[row,col] - observation[row,col])
		else: # upper line (ul)
			z = abs(rmrf[row,col] - rmrf[row,col-1]) + abs(rmrf[row,col] - rmrf[row,col+1]) + abs(rmrf[row,col] - rmrf[row+1,col]) + lamda*abs(rmrf[row,col] - observation[row,col])

	elif(row == (sh[0]-1)):
		if(col == 0):	# bottom-left corner (blc)
			z = abs(rmrf[row,col] - rmrf[row,col+1]) + abs(rmrf[row,col] - rmrf[row-1,col]) + lamda*abs(rmrf[row,col] - observation[row,col])
		elif(col == (sh[1]-1)): # botton-right corner (brc)
			z = abs(rmrf[row,col] - rmrf[row,col-1]) + abs(rmrf[row,col] - rmrf[row-1,col]) + lamda*abs(rmrf[row,col] - observation[row,col])
		else: # bottom line (bl)
			z = abs(rmrf[row,col] - rmrf[row,col-1]) + abs(rmrf[row,col] - rmrf[row,col+1]) + abs(rmrf[row,col] - rmrf[row-1,col]) + lamda*abs(rmrf[row,col] - observation[row,col])
	else:
		if(col == 0):	# left line (ll)
			z = abs(rmrf[row,col] - rmrf[row+1,col]) + abs(rmrf[row,col] - rmrf[row-1,col]) + abs(rmrf[row,col] - rmrf[row,col+1]) + lamda*abs(rmrf[row,col] - observation[row,col])
		elif(col == (sh[1]-1)):	# right line (rl)
			z = abs(rmrf[row,col] - rmrf[row+1,col]) + abs(rmrf[row,col] - rmrf[row-1,col]) + abs(rmrf[row,col] - rmrf[row,col-1]) + lamda*abs(rmrf[row,col] - observation[row,col])
		else:	# rest
			z = abs(rmrf[row,col] - rmrf[row+1,col]) + abs(rmrf[row,col] - rmrf[row-1,col]) + abs(rmrf[row,col] - rmrf[row,col+1]) + abs(rmrf[row,col] - rmrf[row,col-1]) + lamda*abs(rmrf[row,col] - observation[row,col])

	return z


class RMRFwO(RMRF):
	"""
	Implementation of Regular Markov Random Fields (RMRF)
	"""

	#def __init__(self, states=None, observations=None, prior=None, transition=None, observation=None ):
	def __init__(self, states, rmrf, observation, order=1, potencials = None ):
		"""
		Constructor of (R)egular (M)arkov (R)andom (F)ields

		states : 		set of states
		observation :	object with the observation
 		rmrf 	:		numpy matrix of shape (x,y) with initial values
		order	:		useless
		potencials :	useless
		"""

		try:
			spl = states.split("-")
			self.range = True
		except:
			self.range = False			

		if(self.range):
			self.smin = int(spl[0])
			self.smax = int(spl[1])
			self.nstates = self.smax - self.smin
			if( self.nstates < 2):
				raise NameError("Error: The range has to contain at least two states")
		else:
			self.nstates = len(states)
			if(self.nstates < 2):
				raise NameError("Error: The set of states has to contain at least two states.")
				
		self.states = states	# set of states or range

		self.shrmrf = np.shape(rmrf)
		if(len(self.shrmrf) != 2):
			raise NameError("Error: rmrf has to be a numpy matrix of shape (x,y)")
		self.rmrf = rmrf.copy()
		self.shobservation = np.shape(observation)
		if(len(self.shobservation) != 2):
			raise NameError("Error: observation has to be a numpy matrix of shape (x,y)")
		self.observation = observation.copy()


		self.potencials = potencials



	def ICM(self, Uf=smoothingImages, threshold=0.01, maxIterations=10, variant="MAP"):
		raux = self.rmrf.copy()

		for it in range (maxIterations):
			for i in range(self.shrmrf[0]):		#rows
				for j in range(self.shrmrf[1]):	#cols
					t =  self.getDifferentValue(self.rmrf[i,j])	#alternative value
					ufi = Uf(self.rmrf,self.observation,i,j)	#  with position, son only evaluate the local point of interest
					aux = self.rmrf[i,j]
					self.rmrf[i,j] = t				
					uft = Uf(self.rmrf,self.observation,i,j)	#  with position, so only evaluate the local point of interest
					if(uft >= ufi):
						self.rmrf[i,j] = aux # return original value

			if(np.all(raux == self.rmrf)):	#if there are no change, then finish
				print("Succesfully finish, iteration: "+str(it)) 
				break
			else:
				raux = self.rmrf.copy()
		return self.rmrf


	def metropolis(self, Uf=smoothingImages, threshold=0.01, maxIterations=10, prob=0.01, variant="MAP"):
		raux = self.rmrf.copy()

		for it in range (maxIterations):
			for i in range(self.shrmrf[0]):		#rows
				for j in range(self.shrmrf[1]):	#cols
					t =  self.getDifferentValue(self.rmrf[i,j])	#alternative value
					ufi = Uf(self.rmrf,self.observation,i,j)	#  with position, son only evaluate the local point of interest
					aux = self.rmrf[i,j]
					self.rmrf[i,j] = t				
					uft = Uf(self.rmrf,self.observation,i,j)	#  with position, so only evaluate the local point of interest
					if(uft >= ufi):
						if(np.random.rand() > prob):
							self.rmrf[i,j] = aux # return original value

			if(np.all(raux == self.rmrf)):	#if there are no change, then finish
				print("Succesfully finish, iteration: "+str(it)) 
				break
			else:
				raux = self.rmrf.copy()
		return self.rmrf

	"""
	def getDifferentValue(self,value):
		while True:
			if(self.range):			
				nv = np.random.randint(self.smin,self.smax)	#a random value
			else:
				nv = self.states[ np.random.randint(self.nstates) ]	#a random value
			if(nv != value):
				return nv
	"""

	def sAnnealing(self, Uf=smoothingImages, threshold=0.01, maxIterations=10, Temp=0.9, tempReduction=0.7, optimal="MAP"):
		raux = self.rmrf.copy()

		if(optimal == "MPM"):
			#create the variable which contains the number that each state is used.
			mpm = np.array( [[{} for i in range(self.shrmrf[1])] for j in range(self.shrmrf[0]) ] )
			mpm = self.functionMPM(mpm, raux)
		elif(optimal != "MAP"):
			raise NameError("Error, optimal only can take the values 'MPM' or 'MAP'") 
			

		for it in range (maxIterations):
			for i in range(self.shrmrf[0]):		#rows
				for j in range(self.shrmrf[1]):	#cols
					t =  self.getDifferentValue(self.rmrf[i,j])	#alternative value
					ufi = Uf(self.rmrf,self.observation,i,j)	#  with position, son only evaluate the local point of interest
					aux = self.rmrf[i,j]
					self.rmrf[i,j] = t				
					uft = Uf(self.rmrf,self.observation,i,j)	#  with position, son only evaluate the local point of interest
					if(uft >= ufi):
						d = uft - ufi
						pt = np.power(np.e,-d/Temp)	# probability of keeping the value with higher energy
						if(np.random.rand() > pt):
							self.rmrf[i,j] = aux # return original value

			if(np.all(raux == self.rmrf)):	#if there are no change, then finish
				print("Succesfully finish, iteration: "+str(it)) 
				break
			else:
				raux = self.rmrf.copy()
			# the last iteration is not counted
			if(optimal == "MPM"):
				mpm = self.functionMPM(mpm, raux)

		if(optimal == "MPM"):
			return self.returnMPM(mpm)
		else:
			return self.rmrf


	def inference(self, Uf=smoothingImages, maxIterations=10, Temp=1.0, tempReduction=1.0, optimal="MAP"):
		"""
		General method, 
		Temp == 1, and tempReduction == 1, ICM
		Temp != 1, and tempReduction == 1, metropolis
		Temp != 1, and tempReduction != 1, simulated anealing

		"""
		raux = self.rmrf.copy()

		if(optimal == "MPM"):
			#create the variable which contains the number that each state is used.
			mpm = np.array( [[{} for i in range(self.shrmrf[1])] for j in range(self.shrmrf[0]) ] )
			mpm = self.functionMPM(mpm, raux)
		elif(optimal != "MAP"):
			raise NameError("Error, optimal only can take the values 'MPM' or 'MAP'") 
			

		for it in range (maxIterations):
			for i in range(self.shrmrf[0]):		#rows
				for j in range(self.shrmrf[1]):	#cols
					t =  self.getDifferentValue(self.rmrf[i,j])	#alternative value
					ufi = Uf(self.rmrf,self.observation,i,j)	#  with position, son only evaluate the local point of interest
					aux = self.rmrf[i,j]
					self.rmrf[i,j] = t				
					uft = Uf(self.rmrf,self.observation,i,j)	#  with position, son only evaluate the local point of interest
					if(uft >= ufi):
						if(tempReduction == 1):
							if(Temp == 1):	# ICM
								self.rmrf[i,j] = aux # return original value								
							else:			# Metropilis
								if(np.random.rand() > Temp):
									self.rmrf[i,j] = aux # return original value
						else:				#simulated annealing
							d = uft - ufi						
							pt = np.power(np.e,-d/Temp)	# probability of keeping the value with higher energy
							if(np.random.rand() > pt):
								self.rmrf[i,j] = aux # return original value

			if(np.all(raux == self.rmrf)):	#if there are no change, then finish
				print("Succesfully finish, iteration: "+str(it)) 
				break
			else:
				raux = self.rmrf.copy()
			# the last iteration is not counted
			if(optimal == "MPM"):
				mpm = self.functionMPM(mpm, raux)

			Temp *= tempReduction

		if(optimal == "MPM"):
			return self.returnMPM(mpm)
		else:
			return self.rmrf





