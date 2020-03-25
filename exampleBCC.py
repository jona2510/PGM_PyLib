"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np
import PGM_PyLib.BCC as bcc
import PGM_PyLib.naiveBayes as nb

np.random.seed(0)   # it is not necessary
# 5 variable classes
# three classes for each variable class
# 7 attributes

# 300 instances for training
data_train = np.random.randint(0,5,size=(300,7))
cl_train = np.random.randint(0,3,size=(300,5))
# 100 instances for testing
data_test = np.random.randint(0,5,size=(100,7))
cl_test = np.random.randint(0,3,size=(100,5)) 

# create the classifiers 
c = bcc.BCC(chainType="parents", baseClassifier=nb.naiveBayes(), structure="auto")
# train the classifier
c.fit(data_train, cl_train)
# predict 
p = c.predict(data_test)
# evaluation
print(c.exactMatch(cl_test, p))

# show the structure
print(c.structure)
