"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np 

class MDP:
	"""
	Implementation of Markov Decision Processes
	"""

	def __init__(self, reward, stateTransition, discountFactor=0.9 ):
		"""
		Constructor of (M)arkov (D)ecision (P)rocess

		reward 	: 	rewards associated to each state. A ndarray (matrix) of shape (n_states), 
						where each position/state has associated a dictionary, for each item the key is the action that can be taken from the current state and the value is the reward
		stateTransitions : list of size (n_states), each one has a dictionary 
							where in each item, the key is a "neighbour" state 
							and the value is a dictionary where for each item the key is the action and the value is the probability
 nd array of size (n_actions).
		"""

		self.reward = reward.copy()
		self.stateTransition = stateTransition.copy()
		self.discountFactor = discountFactor

		sh = np.shape(self.reward)
		self.nStates = sh[0]		#nStates
		#self.nActions = sh[1]		#nActions

		# 
		self.policy = None

	def valueItetration(self, threshold, maxIter=-1):
		"""
		The value iteration algorithm is applied in order to obtain the optimal policy
		"""

		self.policy = np.zeros(self.nStates)
		# initialization
		V0 = np.zeros(self.nStates)
		for i in range(self.nStates):
			V0[i] = max(self.reward[i].values() )

		Vt = np.zeros(self.nStates) - np.inf

		#print("V0: ",V0)
		#print("Vt: ",Vt)

		# iterative improvement		
		t = 0
		while True:
			for s in range(self.nStates):
				m = - np.inf	# max
				#for a in range(self.nActions):
				for a in self.reward[s].keys():
					ac = 0.0

					for sp in self.stateTransition[s].keys():
						#print("stateTransition[s][sp][a]: ",self.stateTransition[s][sp][a])
						#print("s: ",s)
						#print("a: ",a)
						#print("sp: ",sp)
						#print("V0[sp]: ",V0[int(sp)])
						ac += self.stateTransition[s][sp][a] * V0[sp]
					ac = self.discountFactor * ac + self.reward[s][a]

					if(ac > m):
						m = ac
						self.policy[s] = a

				Vt[s] = m

			t += 1
			# check if the policy has converged
			#print("iter: ",t)
			#print("V0: ",V0)
			#print("Vt: ",Vt)
			#print("**********************************************")
			if(np.all(abs(Vt - V0) < threshold )):
				print("converged!!, number of iterations: " + str(t))
				break
			else:
				V0 = Vt.copy()

			if(maxIter < 0):
				continue
			else:
				if(t >= maxIter):
					print("not converged!!, after " + str(t) +" iterations")
					break


		return self.policy


	def policyItetration(self, maxIter=-1):
		"""
		The value iteration algorithm is applied in order to obtain the optimal policy
		"""

		self.policy = np.zeros(self.nStates)
		policyOld = np.zeros(self.nStates)-1 # save the old policy

		# initialization
		V0 = np.zeros(self.nStates)
		for i in range(self.nStates):
			V0[i] = max(self.reward[i].values() )

		Vt = np.zeros(self.nStates) - np.inf

		#print("policyOld: ",policyOld)
		#print("policy: ",self.policy)

		# iterative improvement		
		t = 0
		while True:
			for s in range(self.nStates):
				m = - np.inf	# max
				#for a in range(self.nActions):
				for a in self.reward[s].keys():
					ac = 0.0

					for sp in self.stateTransition[s].keys():
						#print("stateTransition[s][sp][a]: ",self.stateTransition[s][sp][a])
						#print("sp: ",sp)
						#print("V0[sp]: ",V0[int(sp)])
						ac += self.stateTransition[s][sp][a] * V0[sp]
					ac = self.discountFactor * ac + self.reward[s][a]

					if(ac > m):
						m = ac
						self.policy[s] = a

				Vt[s] = m

			t += 1
			# check if the policy has converged
			#print("iter: ",t)
			#print("policyOld: ",policyOld)
			#print("policy: ",self.policy)
			#print("**********************************************")
			if(np.all( self.policy == policyOld ) ):
				print("converged!!, number of iterations: " + str(t))
				break
			else:
				V0 = Vt.copy()
				policyOld = self.policy.copy()


			if(maxIter < 0):
				continue
			else:
				if(t >= maxIter):
					print("not converged!!, after " + str(t) +" iterations")
					break

		#print("number of iteration: " + str(t))

		return self.policy




