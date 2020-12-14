"""
This code-template belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import numpy as np

class my_ci_test:

	def __init__(self, significance=0.05):
		"""
		Constructor of the class.
			It is highly recommended that you describe the arguments/parameters

			significance : statistical significance
		"""	
		#first check that all parameters are valid	
		#	for example, significance can take the values in  (0,1)
		if( not (0 < significance < 1) ):
			raise NameError("ERROR!!!, significance only can take values in (0,1) ")			

		# parameters
		self.significance = significance

	def test(self,X,Y,Z):
		"""
		Return the result of applying the test I(X,Y|Z), that is, True or False
		"""
		# first check that all parameters are valid (X,Y,Z)
		shz = np.shape(Z)
		if(len(shz) != 2):
			raise NameError("Z has to be a ndarray of shape (n_samples ,m_variables)")
		if( (len(X) != len(Y)) or (len(Y) != shz[0]) ):
			raise NameError("The size of the vectors are different")

		#################################################################################
		#	you can write the code of your conditional indepence test in this section	#
		#################################################################################

		# result_test has the result of the test (True or False)
		return result_test


