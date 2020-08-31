"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np
import PGM_PyLib.HMM as hmm

states = ["M1", "M2"]
obs = ["H", "T"]
PI = np.array( [0.5, 0.5] )	#prior probabilities
A = np.array( [[0.5, 0.5], [0.5, 0.5]] ) #transition probabilities
B = np.array( [[0.8, 0.2], [0.2, 0.8]] ) #observation probabilities

# Inializating the model with all its parameters
h = hmm.HMM(states=states, observations=obs, prior=PI, transition=A, observation=B)

O = ["H","H","T","T"] # observation sequence

# evaluating an observation sequence
print("Score of: H,H,T,T")
print(h.forward(O))

# obtaining the most probable state at each time t
lmps = [h.MPS(i,O) for i in range(1, len(O)+1) ]
print("Most probable state at each time t:")
print(lmps)

# obtaining the most probable sequence of states
mpss,score = h.viterbi(O)
print("Most probable sequence of states:")
print(mpss)

