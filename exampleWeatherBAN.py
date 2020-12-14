import numpy as np
import PGM_PyLib.augmented as abc

data_train = np.array([
	["Sunny","High","High","False"],
	["Sunny","High","High","True"],
	["Overcast","High","High","False"],
	["Rain","Medium","High","False"],
	["Rain","Low","Normal","False"],
	["Rain","Low","Normal","True"],
	["Overcast","Low","Normal","True"],
	["Sunny","Medium","High","False"],
	["Rain","Medium","Normal","False"],
	["Sunny","Medium","Normal","True"],
	["Overcast","Medium","High","True"],
	["Overcast","High","Normal","False"]
])
cl_train = np.array(["No","No","Yes","Yes","Yes",
	"No","Yes","No","Yes","Yes","Yes","Yes"])

c = abc.augmentedBC(algStructure="auto", 
	smooth=0.1, usePrior=True)

c.fit(data_train, cl_train)

data_test = np.array([
	["Rain","Medium","High","True"],
	["Sunny","Low","Normal","False"]
])

p = c.predict(data_test) 
print("Predictions:")
print(p)

print("Structure:")
print(c.structure)


