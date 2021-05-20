Using custom classifiers
===============
The *vanilla* implementation of autoBOTLib uses the *SGDClassifier* class, suitable for fast exploration of a wide array of various models. However, should you wish to use your custom, sklearn-syntax compatible classifier, the following snippet is a good start.


.. code-block:: python

	import autoBOTLib
	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns
	from sklearn.linear_model import SGDClassifier

	## Load example data frame
	dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t")
	train_sequences = dataframe['text_a'].values.tolist()
	train_targets = dataframe['label'].values

	## The syntax for specifying a learner and the hyperparameter space!
	## These are the hyperparameters to be explored for each representation.
	classifier_hyperparameters = {
		"loss": ["hinge"],
		"penalty": ["elasticnet"],
		"alpha": [0.01, 0.001],
		"l1_ratio": [0, 0.001,1]
	}

	## This is the classifier compatible with the hyperparameters.
	custom_classifier = SGDClassifier()

	autoBOTLibObj = autoBOTLib.GAlearner(
		train_sequences,  # input sequences
		train_targets,  # target space 
		time_constraint=0.1,  # time in hours
		num_cpu=4,  # number of CPUs to use
		task_name="example test",  # task identifier
		hof_size=3,  # size of the hall of fame
		top_k_importances=25,  # how many top features to output as final ranking
		memory_storage="../memory",
		representation_type="symbolic",
		learner = custom_classifier,
		learner_hyperparameters = classifier_hyperparameters)  # or neurosymbolic or neural

	autoBOTLibObj.evolve(
		nind=10,  ## population size
		strategy="evolution",  ## optimization strategy
		crossover_proba=0.6,  ## crossover rate
		mutpb=0.4)  ## mutation rate
