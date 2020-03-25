"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import PGM_PyLib.hierarchicalClassification.BNCC as bncc
from sklearn.ensemble import RandomForestClassifier as rfc

# predit with HCP
p1 = bncc.HCP("D_EA_01_FD_b_train_head.arff", "D_EA_01_FD_b_train_data.arff", "D_EA_01_FD_b_test_data.arff", "SP", rfc())
print(p1)

#predict with HCA
p2 = bncc.HCA("D_EA_01_FD_b_train_head.arff", "D_EA_01_FD_b_train_data.arff", "D_EA_01_FD_b_test_data.arff", "SP", rfc())
print(p2)

#predict with HCC
p3 = bncc.HCC("D_EA_01_FD_b_train_head.arff", "D_EA_01_FD_b_train_data.arff", "D_EA_01_FD_b_test_data.arff", "SP", rfc())
print(p3)

#predict with HBA
p4 = bncc.HBA("D_EA_01_FD_b_train_head.arff", "D_EA_01_FD_b_train_data.arff", "D_EA_01_FD_b_test_data.arff", "SP", rfc())
print(p4)
