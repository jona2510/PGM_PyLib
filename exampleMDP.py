"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np
from PGM_PyLib.MDP import MDP 

R = np.array([
	{1:-1, 2:-1},			#0
	{2:-1, 3:-1},
	{1:-1, 2:100, 3:-1},	#2
	{1:-100, 3:-1},		
	{0:-1, 1:-1},			#4
	{0:-1, 1:-1, 2:-100},
	{0:100, 1:-1, 3:-1},	#6
	{0:-1, 2:-1},
	{2:-1, 3:-1},			#8
	{0:-1, 2:-1, 3:-1},
	{0:-100, 3:-1},			#10
])


FI = [	
	{	#   u,  d,  r,  l
		0: {1:0.1, 2:0.1},
		1: {1:0.1, 2:0.8},
		4: {1:0.8, 2:0.1}
	},
	{
		0: {2:0.1, 3:0.8},
		1: {2:0.1, 3:0.1},
		2: {2:0.8, 3:0.1}
	},
	{
		1: {1:0.1, 2:0.1, 3:0.7},
		2: {1:0.1, 2:0.1, 3:0.1},
		3: {1:0.1, 2:0.7, 3:0.1},
		5: {1:0.7, 2:0.1, 3:0.1}
	},
	{
		2: {1:0.1, 3:0.8},
		3: {1:0.1, 3:0.1},
		6: {1:0.8, 3:0.1}
	},
	{
		0: {0:0.8, 1:0.1},
		4: {0:0.1, 1:0.1},
		7: {0:0.1, 1:0.8}
	},
	{
		2: {0:0.7, 1:0.1, 2:0.1},
		5: {0:0.1, 1:0.1, 2:0.1},
		6: {0:0.1, 1:0.1, 2:0.7},
		9: {0:0.1, 1:0.7, 2:0.1}
	},
	{
		3: {0:0.7, 1:0.1, 3:0.1},
		5: {0:0.1, 1:0.1, 3:0.7},
		6: {0:0.1, 1:0.1, 3:0.1},
		10:{0:0.1, 1:0.7, 3:0.1}
	},
	{
		4: {0:0.8, 2:0.1},
		7: {0:0.1, 2:0.1},
		8: {0:0.1, 2:0.8}
	},
	{
		7: {2:0.1, 3:0.8},
		8: {2:0.1, 3:0.1},
		9: {2:0.8, 3:0.1}
	},
	{
		5: {0:0.7, 2:0.1, 3:0.1},
		8: {0:0.1, 2:0.1, 3:0.7},
		9: {0:0.1, 2:0.1, 3:0.1},
		10:{0:0.1, 2:0.7, 3:0.1}
	},
	{
		6: {0:0.8, 3:0.1},
		9: {0:0.1, 3:0.8},
		10:{0:0.1, 3:0.1}
	}
]

mdp = MDP( reward=R, stateTransition=FI, discountFactor=0.9 )

print("value iteration:")
policy = mdp.valueItetration(0.1)
print("policy:\n",policy)

print("\n policy iteration:")
policy = mdp.policyItetration()
print("policy:\n",policy)









