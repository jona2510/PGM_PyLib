"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np
import PGM_PyLib.HMM as hmm

data=[["H", "H", "T", "H", "T", "H", "T", "H", "T", "T"],
	["T", "H", "H", "T", "T"],
	["T", "H", "H", "T", "T", "H", "T"],
	["H", "T", "T", "T", "T", "H", "T", "T", "T"],
	["T", "T", "T", "H", "T", "T"]]
h = hmm.HMM()	# empty model

# learning the model from data
h.learn(data,tol=0.001,hs=2,max_iter=100,initialization="random",seed=0)

print("Set of states:")
print(h.states)
print("Set of observations:")
print(h.obs)
print("Prior probabilities")
print(h.prior)
print("Transition probabilities")
print(h.transition)
print("Observation probabilities")
print(h.observation)

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

