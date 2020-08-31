"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np
import PGM_PyLib.naiveBayes as nb

np.random.seed(0)   # it is not necessary
# two classes
# 5 attributes: 2 nominal and 3 numeric

# 200 instances for training
data_train = np.random.randint(0,5,size=(200,2))
data_train = np.concatenate([data_train, np.random.rand(200,3)],axis=1)
cl_train = np.random.randint(0,2,size=200)

# 100 instances for testing
data_test = np.random.randint(0,5,size=(100,2))
data_test = np.concatenate([data_test, np.random.rand(100,3)],axis=1)
cl_test = np.random.randint(0,2,size=100) 

# create the dictionary with the values that each attribute can take
values = {0: [0,1,2,3,4], 1: [0,1,2,3,4], 2:"numeric", 3:"numeric", 4:"numeric"}


# Example 1, a GNB classifier is trained providing "values"
print("Results of GNBC:")
# create the classifiers 
c = nb.GaussianNaiveBayes(smooth=0.1, usePrior=True, meta=values)
# train the classifier
c.fit(data_train, cl_train)
# predict 
p = c.predict(data_test)
# evaluation
print(c.exactMatch(cl_test, p))

# ignore the Prior probabilities
c.usePrior = False
p = c.predict(data_test)
print(c.exactMatch(cl_test,p))


# Example 2, a GNB classifier is trained considering all attributes to be numeric
print("Results of GNBC considering all attributes to be numeric:")
# create the classifiers 
c2 = nb.GaussianNaiveBayes(smooth=0.1, usePrior=True, meta="")
# train the classifier
c2.fit(data_train, cl_train)
# predict 
p = c2.predict(data_test)
# evaluation
print(c2.exactMatch(cl_test, p))

# ignore the Prior probabilities
c2.usePrior = False
p = c2.predict(data_test)
print(c2.exactMatch(cl_test,p))


