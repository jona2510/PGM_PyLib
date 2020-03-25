"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np
import PGM_PyLib.structures.trees as trees

np.random.seed (0)  # it is not necessary

# 7 variables 
# 200 instances 
data = np.random.randint(0,4,size=(200,3))
data = np.concatenate([data, np.random.randint(2,6,size=(200,4))],axis=1)

# create a instance of CLP_MI
clp = trees.CLP_MI(root=0, heuristic=False, smooth=0.1)
# create the structure
structure = clp.createStructure(data)
# show structure
print(structure)
print()

# Use heuristic to automatically select the root of the tree
clp.heuristic = True
structure = clp.createStructure(data)
#show structure
print(structure)

#show the root node of the tree
print(clp.root)

