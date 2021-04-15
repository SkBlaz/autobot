Model persistence
===============
We next demonstrate how simple it is to load a pre-trained model and obtain some predictions. The example assumes you are in the `./examples` folder of the repo.

.. code:: python3

    import autoBOTLib
    import pandas as pd

    ## Load example data frame
    dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t")
    train_sequences = dataframe['text_a'].values.tolist()
    train_targets = dataframe['label'].values

    autoBOTLibObj = autoBOTLib.GAlearner(
	train_sequences,  # input sequences
	train_targets,  # target space 
	time_constraint=2,  # time in hours
	num_cpu="all",  # number of CPUs to use
	task_name="example test",  # task identifier
	hof_size=3,  # size of the hall of fame
	top_k_importances=25,  # how many top features to output as final ranking
	memory_storage=
	"../memory/conceptnet.txt.gz",  # tripled base for concept features
	representation_type="symbolic")  # or symbolic or neural

    autoBOTLibObj.evolve(
       nind=8,  ## population size
       strategy="evolution",  ## optimization strategy
       crossover_proba=0.6,  ## crossover rate
       mutpb=0.4)  ## mutation rate

    ## Persistence demonstration (how to store models for further use?)
    autoBOTLib.store_autobot_model(autoBOTLibObj, "../stored_models/example_insults_model.pickle")


Let's next load the very same model and do some predictions.
    
.. code:: python3

	  
    ## A simple example showcasing the minimal usecase of autoBOTLib on an insults classification data.

    import autoBOTLib
    import pandas as pd

    ## Simply load the model
    autoBOTLibObj = autoBOTLib.load_autobot_model("../stored_models/example_insults_model.pickle")
    dataframe2 = pd.read_csv("../data/insults/test.tsv", sep="\t")
    test_sequences = dataframe2['text_a'].values.tolist()
    test_targets = dataframe2['label'].values

    ## Predict with the model
    predictions = autoBOTLibObj.predict(test_sequences)
    performance = autoBOTLib.compute_metrics(
	"first_run_task_name", predictions,
	test_targets)  ## compute F1, acc and F1_acc (as in GLUE)

    print(performance)
