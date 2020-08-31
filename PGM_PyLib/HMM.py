"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np 

class HMM:
	"""
	Implementationf of Hiden Markov Models (HMM)
	"""

	def __init__(self, states=None, observations=None, prior=None, transition=None, observation=None ):
		"""
		Constructor of HMM

		states : 		set of states
		observation :	set of observtions
		prior :			prior probabilities ndarray of shape (number_states,) 
		transition :	transition probabilities, ndarray of shape (number_states, number_states)
		observation :	observation probabilities, ndarray of shape (number_states, number_observations)
		"""

		self.states = states 				# set of states
		self.obs = observations				# set of observtions
		self.prior = prior					# prior probabilities ndarray of shape (number_states,) 
		self.transition = transition		# transition probabilities, ndarray of shape (number_states, number_states)
		self.observation = observation		# observation probabilities, ndarray of shape (number_states, number_observations)

		self.dicStates = {}
		self.dicObs = {}

		self.configured = False				# if configured, the different methods can be used

		if(self.areParametersCorrect()):
			self.createAuxVariables()
		else:
			print("Empty model")


	def createAuxVariables(self):
		"""
		this method is called after checkError().
		It creates some auxiliar variables, which are used in different methods
		"""		
		for i in range(len(self.states)):
			self.dicStates[self.states[i]] = i

		for i in range(len(self.obs)):
			self.dicObs[self.obs[i]] = i
		
		self.configured = True


	def forward_t(self, t, O):
		"""
		The Forward algorithm is applied from 1 up to t		(0, t-1)
		O : is the obsevation sequence, that is, a list, p.e. [1,2,2,1,3]
		"""
		if(len(np.shape(O)) != 1):
			raise NameError("Error, the observation sequence has to be a list, p.e. [1,2,1,2]")
		if(t < 1):
			raise NameError("Error, t has to be greater or equal than 1")

		N = len(self.states)	#number of states
		alfa = np.zeros(N)

		# initialization
		for i in range(N):		#walks over the states
			alfa[i] = self.prior[i] * self.observation[ i, self.dicObs[ O[0] ] ]	

		# induction
		#walk over the time (observations)
		for i in range(1, t):
			alfa_new = alfa.copy()
			#walk over the states
			for j in range(N):
				aux=0				
				#walk over the states
				for k in range(N):
					aux += alfa[k] * self.transition[k,j]
						
				alfa_new[j] = aux * self.observation[j, self.dicObs[ O[i] ] ]
			alfa = alfa_new
		
		return alfa

	def forward(self, O):
		"""
		The Forward algorithm is applied to the full sequence of observations
		O : is the obsevation sequence, that is, a list, p.e. [1,2,2,1,3]
		"""
		if(len(np.shape(O)) != 1):
			raise NameError("Error, the observation sequence has to be a list, p.e. [1,2,1,2]")

		#first call forward_t
		alfa = self.forward_t(len(O),O)

		# Termination 
		return np.sum(alfa)

	def backward_t(self, t, O):
		"""
		The Backward algorithm is applied from T up to t		(0, t-1)
		O : is the obsevation sequence, that is, a list, p.e. [1,2,2,1,3]
		"""
		if(len(np.shape(O)) != 1):
			raise NameError("Error, the observation sequence has to be a list, p.e. [1,2,1,2]")
		if(t < 1):
			raise NameError("Error, t has to be greater or equal than 1")

		N = len(self.states)	#number of states

		#initialization
		beta = np.zeros(N)+1	# time T
		
		#induction
		#walk over the time (observations)
		for i in range(len(O)-1, t-1, -1):
			beta_new = beta.copy()
			#walk over the states
			for j in range(N):
				aux=0				
				#walk over the states
				for k in range(N):
					aux += beta[k] * self.transition[j,k] * self.observation[k, self.dicObs[ O[i] ] ]
						
				beta_new[j] = aux #* self.observation[j, self.dicObs[ O[i] ] ]
			beta = beta_new
		
		return beta


	def backward(self, O):
		"""
		The Backward algorithm is applied to the full sequence of observations
		O : is the obsevation sequence, that is, a list, p.e. [1,2,2,1,3]
		"""
		raise NameError("backward is not working yet, instead you could use backward_t()")

		if(len(np.shape(O)) != 1):
			raise NameError("Error, the observation sequence has to be a list, p.e. [1,2,1,2]")

		#first call backward_t
		beta = self.backward_t(1,O)

		N = len(self.states)
		# Termination 
		aux=0			
		#walk over the states
		for j in range(N):	
			#walk over the states
			for k in range(N):
				aux += beta[k] * self.transition[j,k] * self.observation[k, self.dicObs[ O[0] ] ]

		return aux

	def for_back(self,O):
		"""
		It applies forward and backward
		this is only a simulation, a better way is to implemente this function in parallel
		"""
		m = int(len(O)/2)
		#first forward
		alfa = self.forward_t(m,O)

		#then backward
		beta = self.backward_t(m,O)

		#join the probabilities of each one
		res = 0
		for i in range(len(self.states)):
			res += alfa[i] * beta[i]

		return res
	
	def gamma(self, t, O):
		"""
		obtain the probability of each state at time t
		O is the observation sequence
		"""
		if(len(np.shape(O)) != 1):
			raise NameError("Error, the observation sequence has to be a list, p.e. [1,2,1,2]")
		if(t < 1):
			raise NameError("Error, t has to be greater or equal than 1")
		
		#first forward
		alfa = self.forward_t(t,O)

		#then backward
		beta = self.backward_t(t,O)

		#join the probabilities of each one
		res = np.zeros(len(self.states))
		for i in range(len(self.states)):
			res[i] = alfa[i] * beta[i]

		return (res/np.sum(res))


	def MPS(self, t, O):
		"""
		Most Probable State (MPS) at time t
		obtain the probability of each state at time t
		O is the observation sequence
		"""
		gammas = self.gamma(t, O)

		pos = np.where( gammas == np.max(gammas) )[0][0]

		return self.states[pos]

	def viterbi(self, O):
		"""
		The Viterbi algorithm is applied
		O : is the obsevation sequence, that is, a list, p.e. [1,2,2,1,3]
		"""
		if(len(np.shape(O)) != 1):
			raise NameError("Error, the observation sequence has to be a list, p.e. [1,2,1,2]")

		N = len(self.states)	#number of states
		sigma = np.zeros(N)
		fi = np.zeros( (len(O)-1, N) ).astype(int)

		aux = np.zeros(N)

		# initialization
		for i in range(N):		#walks over the states
			sigma[i] = self.prior[i] * self.observation[ i, self.dicObs[ O[0] ] ]

		# induction
		#walk over the time (observations)
		for i in range(1, len(O)):
			#fi.append([])			# maybe a matrix is a better way, cause its length is equal to length of observation sequence  (O)
			sigma_new = sigma.copy()
			#walk over the states
			for j in range(N):
				#aux = 0				
				#walk over the states
				#for k in range(N):
				aux2 = sigma * self.transition[:,j]
				#print(aux2)
				posmax = np.where( aux2 == np.max(aux2) )[0][0]
				#print(posmax)

				#fi[i].append( self.states[posmax] )
				fi[i-1, j] = posmax
				sigma_new[j] = aux2[posmax] * self.observation[ j, self.dicObs[ O[i] ] ]

				#aux[k] = np.max(sigma_new)	#alfa[k] * self.transition[k,j]
						
				#alfa_new[j] = aux * self.observation[j, self.dicObs[ O[i] ] ]
			sigma = sigma_new

		#termination
		valmax = np.max(sigma)
		posmax = np.where( sigma == valmax )[0][0]
		rStates = []

		rStates.append( self.states[posmax] )
		for i in range(len(fi)-1,-1,-1):
			posmax = fi[i,posmax]
			rStates.append( self.states[posmax] )
			
		#print("sigma:")
		#print(sigma)
		#print("fi:")
		#print(fi)

		rStates.reverse()

		return (rStates,valmax)

	def evaluation(self):
		"""
		The probability of a sequence of observations is estimated
		"""
		return

	def optimalSequence(self):
		"""
		given an observation sequence, estimate the most probable  state sequence that produced the observations
		"""
		return

	def xi(self, t, O):
		"""
		Estimate xi at time t, for all (i,j)

		t has to be lower than T and greater than 0
		"""
		#first forward
		alfa = self.forward_t(t,O)
		#then backward
		beta = self.backward_t(t+1,O)

		ns = len(self.states)	#number of states
		# a transition states x states
		# b observation states x observation

		# Create a matrix aux
		mxi = np.zeros( (ns,ns) )
		for i in range(ns):
			for j in range(ns):
				# it is 't' cause in the observation sequence, 1 corresponded to the first element and so on
				# and the theory says t+1
				mxi[i,j] = alfa[i] * self.transition[i,j] * self.observation[j, self.dicObs[O[t]] ] * beta[j]

		# obtain and return the 'joint' probabilities, these are not conditional probs.
		return ( mxi/np.sum(mxi) )

	def getObservations(self,data):
		"""
		Obtain the states along data
		"""
		states = set()
		for x in data:
			states = states | set(x)		
		return sorted(states)


	def learn(self,data,tol=0.01,hs=3,max_iter=10,initialization="uniform",seed=0):
		"""
		The Baum-Welch algorithm is applied to learn the parameters

		data : it is a list, which contains lists or ndarrays of shape (x,) with the chains of observations
			p.e.: [ [1,2,3], [2,1,3,2,3], [1,3,2,2,1,2] ]
		hs : number of (h)idden (s)tates
		intialization : how the parameters are initializated, with 'uniform' or 'random' probabilities
		seed : if 'auto' then it uses the seed of numpy, if seed is an int, this is used as the seed
		"""
		
		#########################################################
		#						INITIALIZATION					#
		#########################################################
		self.obs =  self.getObservations(data)
		nObs = len(self.obs)		
		self.states = [ i for i in range(hs)]

		if(initialization == "uniform"):
			self.prior = np.ones( hs )	
			self.transition = np.ones((hs,hs))			# transition probabilities, ndarray of shape (number_states, number_states)
			self.observation = np.ones((hs,nObs))			# observation probabilities, ndarray of shape (number_states, number_observations)
		elif(initialization == "random"):
			if(seed == 'auto'):
				None
			else:
				np.random.seed(seed)
			self.prior = np.random.rand(hs)	
			self.transition = np.random.rand(hs,hs)			# transition probabilities, ndarray of shape (number_states, number_states)
			self.observation = np.random.rand(hs,nObs)		# observation probabilities, ndarray of shape (number_states, number_observations)
		else:
			raise NameError("initialization only can take the values 'uniform' or 'random'")

		# normalize probabilities
		self.prior = self.prior / np.sum(self.prior)
		for i in range(hs):
			self.transition[i] = self.transition[i] / np.sum(self.transition[i])
			self.observation[i] = self.observation[i] / np.sum(self.observation[i])

		self.dicStates = {}
		self.dicObs = {}		
		self.createAuxVariables()

		#########################################################
		#			CYCLE OF EXPECTATION-MAXIMIZATION			#
		#########################################################
		priorAux = np.zeros( hs )	
		transitionAux = np.zeros((hs,hs))			
		observationAux = np.zeros((hs,nObs))
		flag = True
		e = 0	# epoch / era / iteration
				
		while(e < max_iter and flag):
			self.BaumWelch(priorAux, transitionAux, observationAux, data)		

			# if there are significant changes, execute BW again, else break the cycle

			if( np.all( np.abs(self.prior - priorAux) < tol) ):
				if( np.all( np.abs(self.transition - transitionAux) < tol) ):
					if( np.all( np.abs(self.observation - observationAux) < tol) ):
						flag = False
						print("The threshold of (tol)erance was reached!")

			#first check  for significant changes

			# assign auxiliars to the main
			self.prior = priorAux.copy()
			self.transition = transitionAux.copy()
			self.observation = observationAux.copy()

			print("***************************************\niter: " + str(e))
			print("prior:")
			print(self.prior)
			print("transition:")
			print(self.transition)
			print("observation:")
			print(self.observation)

			e += 1


	def BaumWelch(self,prior, transition, observation, data):
		"""
		Executes one iteration of Baum-Welch Algorithm
		data contains n different observation sequences
		"""
		transition.fill(0)		#clean the matrix
		observation.fill(0)		#clean the matrix

		ns = len(self.states)
		no = len(self.obs)

		# walk over the observation sequences
		for O in data:
			#step 1
			prior[:] = self.gamma(1,O)	#start	at time t = 1
		
			#step 2
			for t in range(1,len(O)):	# t : (1,T-1)
				transition[:,:] = transition + self.xi(t,O)		
			
			#step 3
			#for j in range(ns):
			for t in range(len(O)):
				gammaAux = self.gamma(t + 1,O)	# from t=1 up to T
				#for j in range(ns):
				observation[ :, self.dicObs[ O[t] ] ] = observation[ :, self.dicObs[ O[t] ] ] + gammaAux
			
		# normalize probabilities as conditional probabilities
		# prior is already normalized
		# normalize transition and observation
		for i in range(len(transition)):
			transition[i,:] = transition[i] / np.sum( transition[i] )
			observation[i,:] = observation[i,:] / np.sum( observation[i] )


	def checkErrors(self):
		"""
		Check if all the parameters are correctly set
		"""
		#states=None, observations=None, prior=None, transition=None, observation=None
		shSt = np.shape(self.states)
		if( len( shSt ) != 1 ):
			raise NameError("Error, the set of states was not provided correctly")
		if(shSt[0] < 2):
			raise NameError("Two or more states are requiered")			

		shObs = np.shape(self.obs)
		if( len( shObs ) != 1 ):
			raise NameError("Error, the set of observations was not provided correctly")
		if(shObs[0] < 2):
			raise NameError("Two or more observations are requiered")

		shPr = np.shape(self.prior)
		if( len( shPr ) != 1 ):
			raise NameError("Error, the prior probabilities were not provided correctly")

		shTr = np.shape(self.transition)
		if( len( shTr ) != 2 ):
			raise NameError("Error, the transition probabilities were not provided correctly")

		shOb = np.shape(self.observation)
		if( len( shOb ) != 2 ):
			raise NameError("Error, the observation probabilities were not provided correctly")


		## check the size of the matrix are correct
		if(shPr[0] != shSt[0]):
			raise NameError("Error, the number of prior probabilities is different than the number of states")

		if( not (shSt[0] == shTr[0] == shTr[1]) ):
			raise NameError("Error, the shape of transition probabilities matrix has to be (number_states, number_states) ")

		if( (shOb[0] != shSt[0]) or (shOb[1] != shObs[0]) ):
			raise NameError("Error, the shape of the observation probabilities matrix has to be (number_states, number_observations)")


		# probabilities >= 0
		if( not np.all(self.prior>=0) ):
			raise NameError("Error, there are negative values in prior probabilities")

		if( not np.all(self.transition>=0) ):
			raise NameError("Error, there are negative values in transition probabilities")

		if( not np.all(self.observation>=0) ):
			raise NameError("Error, there are negative values in observation probabilities")

		# should the method check if the different probabilities parameters sum 1
		#		the problem is the precision of the computers, that is, hardly it will sum exactly one
		
	def areParametersCorrect(self):
		"""
		Check if all the parameters are correctly set
		"""
		#states=None, observations=None, prior=None, transition=None, observation=None
		shSt = np.shape(self.states)
		if( len( shSt ) != 1 ):
			print("Error, the set of states was not provided correctly")
			return False
		if(shSt[0] < 2):
			print("Two or more states are requiered")			
			return False

		shObs = np.shape(self.obs)
		if( len( shObs ) != 1 ):
			print("Error, the set of observations was not provided correctly")
			return False
		if(shObs[0] < 2):
			print("Two or more observations are requiered")
			return False

		shPr = np.shape(self.prior)
		if( len( shPr ) != 1 ):
			print("Error, the prior probabilities were not provided correctly")
			return False

		shTr = np.shape(self.transition)
		if( len( shTr ) != 2 ):
			print("Error, the transition probabilities were not provided correctly")
			return False

		shOb = np.shape(self.observation)
		if( len( shOb ) != 2 ):
			print("Error, the observation probabilities were not provided correctly")
			return False


		## check the size of the matrix are correct
		if(shPr[0] != shSt[0]):
			print("Error, the number of prior probabilities is different than the number of states")
			return False

		if( not (shSt[0] == shTr[0] == shTr[1]) ):
			print("Error, the shape of transition probabilities matrix has to be (number_states, number_states) ")
			return False

		if( (shOb[0] != shSt[0]) or (shOb[1] != shObs[0]) ):
			print("Error, the shape of the observation probabilities matrix has to be (number_states, number_observations)")
			return False


		# probabilities >= 0
		if( not np.all(self.prior>=0) ):
			print("Error, there are negative values in prior probabilities")
			return False

		if( not np.all(self.transition>=0) ):
			print("Error, there are negative values in transition probabilities")
			return False

		if( not np.all(self.observation>=0) ):
			print("Error, there are negative values in observation probabilities")		
			return False

		return True

