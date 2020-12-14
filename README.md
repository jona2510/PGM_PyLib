# PGM_PyLib: A Python Library for Inference and Learning of Probabilistic Graphical Models


The *Probabilistic Graphical Models Python Library* (**PGM_PyLib**) was written for inference and learning of several classes of Probabilistic Graphical Models (PGM)  in Python. 
The theory behind the different algorithms can be found in the book *Probabilistic Graphical Models Principles and Applications* of Luis Enrique Sucar.

# PGM_PyLib Content 
PGM_PyLib include several algorithms based on a graphical representation of independence relations, such as:
- **Bayesian Classifiers**
	- Naive Bayes Classifier
	- Gaussian Naive Bayes Classifier
	- Bayesian Network augmented Bayesian Classifier (BAN)
	- Semi Naive Bayes Classifier
	- Bayesian Chain Classifier
	- Hierarchical classification with Bayesian Networks and Chained Classifiers
- **Hidden Markov Models**
- **Markov Random Models**
- **Bayesian Networks**
	- Chow-Liu procedure (CLP)
	- CLP with Conditional Mutual information
	- PC algorithm
- **Markov Decision Processes**
   
Please check the manual for the full list of algorithms.

The "PGM_PyLib Manual vX.X.pdf" contains the description of the PGM's which were implemented, also you will find different examples.

# Cite us
If you use the library, please cite us.

From this work, we published the paper *PGM_PyLib: A Toolkit for Probabilistic Graphical Models in Python*:
```
@InProceedings{pgm-pylib,
	title = {PGM{\_}PyLib: A Toolkit for Probabilistic Graphical Models in Python},
	author = {Serrano-⁠P{\'e}rez, Jonathan and Sucar, L. Enrique},
	booktitle = {The 10th International Conference on Probabilistic Graphical Models}, 
	year = 2020,
	month = September,
	address = {Aalborg, Denmark}
}
```
Please, also cite the manual and the book.
```
@manual{pgm-pylib-manual,
	title = {PGM_PyLib: A Python Library for Inference and Learning of Probabilistic Graphical Models},
	author = {Serrano-⁠P{\'e}rez, Jonathan and Sucar, L. Enrique},
	year = 2020
}

@book{pgm-book, 
    author = {L. Enrique Sucar}, year = 2015,
    title = {Probabilistic Graphical Models Principles and Applications }, 
    edition = 1, 
    publisher = {Springer-Verlag London}, 
    adress = {London} 
}
```

