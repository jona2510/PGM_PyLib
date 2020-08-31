"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np
from MRF import RMRFwO as mrf

s = [0,1]
r = np.zeros((4,4),dtype=int)
print("Initial RMRF\n",r)
obs=np.array([[0,0,0,0],[0,1,1,0],[0,1,0,0],[0,0,1,0]])
print("\nObservation\n",obs)


#ICM with MPM
mr = mrf(s,r,obs)
print("\nICM, MPM:")
r = mr.inference(maxIterations=100, Temp=1.0, tempReduction=1.0, optimal="MPM")
print(r)
#Metropolis with MPM
mr = mrf(s,r,obs)
print("\nMetropolis, MPM:")
r = mr.inference(maxIterations=100, Temp=0.01, tempReduction=1.0, optimal="MPM")
print(r)
#simulated annealing with MAP
mr = mrf(s,r,obs)
print("\nSimulated annealing, MAP:")
r = mr.inference(maxIterations=100, Temp=0.9, tempReduction=0.8, optimal="MAP")
print(r)
