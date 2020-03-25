"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np
import PGM_PyLib.augmented as pnd

np.random.seed (0)  # it is not necessary

# 5 variables 
# variables 0 and 1 can take the values [7,8]
# variables 2,3,4 can take the values [10,11,12]
# 100 instances 
data = np.random.randint(7,9,size=(100,3))
data = np.concatenate([data, np.random.randint(10,13,size=(100,2))], axis=1 )

# variables contains the values that each variable can take
variables = {0:[7,8], 1:[7,8], 2:[10,11,12], 3:[10,11,12], 4:[10,11,12]}

# Example 1: we want to estimate P(0|1,2,3,4)
positions = [0,1,2,3,4]
cpt = pnd.probsND(variables, positions, smooth=0.1)
cpt.estimateProbs(data)
#show the conditional probabilities
print(cpt.probabilities)
print("******************************************")

# Example 2: we want to estimate P(3|1,4)
positions = [3,1,4]
cpt2 = pnd.probsND(variables, positions, smooth=0.1)
cpt2.estimateProbs(data)
#show the conditional probabilities
print(cpt2.probabilities)
print()
# The sum of aeA P(a|b,c,...) = 1, forall beB, forall ceC ... 
#Example 3: below has to sum 1
print(cpt2.probabilities[0,0,0] + cpt2.probabilities[1,0,0] + cpt2.probabilities[2,0,0])

#Example 4: below has to sum 1
print(cpt2.probabilities[0,1,1] + cpt2.probabilities[1,1,1] + cpt2.probabilities[2,1,1])
