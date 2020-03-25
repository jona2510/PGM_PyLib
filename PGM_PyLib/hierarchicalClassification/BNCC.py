"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

#import PGMLib.hierarchicalClassification.BNCC as bncc
from sklearn.ensemble import RandomForestClassifier as rfc
import PGM_PyLib.hierarchicalClassification.HBA_pre as hba
import PGM_PyLib.hierarchicalClassification.HCA_pre as hca
import PGM_PyLib.hierarchicalClassification.HCC_pre as hcc
import PGM_PyLib.hierarchicalClassification.HCP_pre as hcp



def HBA(header_in, train_in, test_in, tscore="SP", baseClassifier=rfc(), type_prop="all_probabilities"):
	return hba.HBA(header_in, train_in, test_in, tscore, type_prop, baseClassifier)

def HCA(header_in, train_in, test_in, tscore="SP", baseClassifier=rfc(), type_prop="all_probabilities"):
	return hca.HCA(header_in, train_in, test_in, tscore, type_prop, baseClassifier)

def HCP(header_in, train_in, test_in, tscore="SP", baseClassifier=rfc(), type_prop="all_probabilities"):
	return hcp.HCP(header_in, train_in, test_in, tscore, type_prop, baseClassifier)

def HCC(header_in, train_in, test_in, tscore="SP", baseClassifier=rfc(), type_prop="all_probabilities"):
	return hcc.HCC(header_in, train_in, test_in, tscore, type_prop, baseClassifier)




