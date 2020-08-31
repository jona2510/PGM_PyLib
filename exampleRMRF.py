"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np
from MRF import RMRF as rmrf

np.random.seed(0)	# no mandatory

s = [i for i in range(6)]	#s = "0-6"
r = np.random.randint(0,6,size=(5,7))	# RMRF of size 5x7
print("initial RMRF\n",r)

#ICM with MAP
mr = rmrf(s,r)
print("\nICM, MAP:")
r = mr.inference(maxIterations=100, Temp=1.0, tempReduction=1.0, optimal="MAP")
print(r)
#Metropolis with MAP
mr = rmrf(s,r)
print("\nMetropolis, MAP:")
r = mr.inference(maxIterations=100, Temp=0.01, tempReduction=1.0, optimal="MAP")
print(r)
#simulated annealing with MPM
mr = rmrf(s,r)
print("\nSimulated annealing, MPM:")
r = mr.inference(maxIterations=100, Temp=0.9, tempReduction=0.8, optimal="MPM")
print(r)
