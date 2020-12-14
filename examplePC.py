"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np
from PGM_PyLib.structures.DAG import PC
#from PGM_PyLib.utils import chi2_test
from PGM_PyLib.stat_tests.ci_test import chi2_ci_test
from scipy.stats import bernoulli as ber
#from scipy.stats import chi2 as c2

nv = 5000
np.random.seed(999999)  # it is not necessary

x = ber.rvs(0.3,size=nv)
z1 = np.zeros( nv )
z2 = np.zeros( nv )
y1 = np.zeros( nv )

for i in range(nv):
	if(x[i]==0):
		z1[i] = ber.rvs(0.3)
		z2[i] = ber.rvs(0.8)
	else:
		z1[i] = ber.rvs(0.4)
		z2[i] = ber.rvs(0.6)
	if(z1[i]==0):
		if(z2[i]==0):	# 0,0
			y1[i] = ber.rvs(0.7)
		else:	#0,1
			y1[i] = ber.rvs(0.1)
	else:
		if(z2[i]==0):	# 1,0
			y1[i] = ber.rvs(0.7)		
		else:	# 1,1
			y1[i] = ber.rvs(0.3)

y2 = np.zeros( nv )
for i in range(nv):
	if(z2[i]==0):	# 0
		y2[i] = ber.rvs(0.25)
	else:	#1
		y2[i] = ber.rvs(0.65)

# the data
# 	five variables, each one can take the values 0-1
data = np.column_stack([x,z1,z2,y1,y2])

# conditional independence tests:
#	tt: learning the underlying undirected graph with a significance of 0.05:
#	td: orient edges of the graph with a significance of 0.3:
tt = chi2_ci_test(significance=0.05, correction=False, lambda_=None,smooth=0.0)
td = chi2_ci_test(significance=0.3, correction=False, lambda_=None,smooth=0.0)

# for ci tests, maximun 3 conditional variables are considered
pct = PC(3, itest=tt, itestDir=td, column_order="original")

# generate structure with data
pct.createStructure(data)
print(pct.structure)
# apply orientation rules for patterns
pct.orientationRules2()

# show the obtained graph
print(pct.structure)


