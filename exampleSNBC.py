"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np
import PGM_PyLib.semiNaive as sn

np.random.seed(0)   # it is not necessary
# three classes
# 5 attributes

# 100 instances for training
data_train = np.random.randint(0,5,size=(100,5)).astype(str)
cl_train = np.random.randint(0,3,size=100)
# 50 instances for testing
data_test = np.random.randint(0,5,size=(50,5)).astype(str)
cl_test = np.random.randint(0,3,size=50) 

# create the classifiers 
c = sn.semiNaive(validation=0.8, epsilon=0.01, omega=0.01, smooth=0.1, nameAtts="auto", usePrior=True)
# train the classifier
c.fit(data_train, cl_train)
# predict 
p = c.predict(data_test)
# evaluation
print(c.exactMatch(cl_test, p))

# ignore the Prior probabilities
c.NBC.usePrior = False
p = c.predict(data_test)
print(c.exactMatch(cl_test,p))

# show the operations that were applied
print(c.opeNameAtts)
