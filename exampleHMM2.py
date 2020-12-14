import numpy as np
import PGM_PyLib.HMM as hmm

states = ["M1", "M2"]
obs = ["H", "T"]
PI = np.array( [0.5, 0.5] )	#prior probabilities
A = np.array( [[0.5, 0.5], [0.5, 0.5]] ) #transition probabilities
B = np.array( [[0.8, 0.2], [0.2, 0.8]] ) #observation probabilities

h = hmm.HMM(states=states, observations=obs, prior=PI, transition=A, observation=B)

O = ["H","H","T","T"] # observation sequence

print("Score of: H,H,T,T")
print(h.forward(O))

print("Most probable sequence of states:")
mpss,score = h.viterbi(O)
print(mpss)


