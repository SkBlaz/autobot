autoBOTLib CLI
===============
To streamline the experiments, it makes a lot of sense to directly use the *autoBOTLib* as a CLI tool. The library itself implements wrappers for main functions, and can be executed as follows. By running

.. code-block:: text

	python autoBOTLib --arguments

The arguments are defined as follows.

.. code-block:: text
				
    usage: autoBOTLib [-h] [--classifier CLASSIFIER] [--time TIME] [--popsize POPSIZE]
		   [--output_folder OUTPUT_FOLDER] [--hof_size HOF_SIZE]
		   [--representation_type REPRESENTATION_TYPE] [--datafolder DATAFOLDER]
		   [--mutation_rate MUTATION_RATE] [--crossover_rate CROSSOVER_RATE]
		   [--predict_data PREDICT_DATA] [--load_model LOAD_MODEL] [--num_cpu NUM_CPU]
		   [--mode MODE]

    optional arguments:
      -h, --help            show this help message and exit
      --classifier CLASSIFIER
      --time TIME
      --popsize POPSIZE
      --output_folder OUTPUT_FOLDER
      --hof_size HOF_SIZE
      --representation_type REPRESENTATION_TYPE
      --datafolder DATAFOLDER
      --mutation_rate MUTATION_RATE
      --crossover_rate CROSSOVER_RATE
      --predict_data PREDICT_DATA
      --load_model LOAD_MODEL
      --num_cpu NUM_CPU
      --mode MODE

The code outputs the main results into the *results* folder by default. 

.. csv-table:: Main hyperparameters
   :header: "Parameter", "Description", "Values"
   :widths: 15, 10, 30

   "--classifier", "Classification engine used SGD (default)", "classifier used"
   "--time","Time available (in hours)", integer
   "--popsize","Population size", integer
   "--output_folder","output folder where the results will be stored",string
   "--hof_size","Size of the hall of fame",integer
   "--representation_type","The type of the representation space to be evolved", "symbolic; neurosymbolic; neural"
   "--datafolder","The folder of the input data", string
   "--mutation_rate",The mutation rate, float [0 to 1]
   "--crossover_rate",The crossover rate, float [0 to 1]
   "--load_model",Path to the stored model, str
   "--predict_data", test instances for which the predictions are to be made, str
   "--num_cpu", Number of parallel learners, int [default 8]
   "--mode", learning or prediction, str ["prediction","learning"]


Note that there are two main modes of operation: learning and prediction. 
An example of the output is given in `results folder. <https://github.com/SkBlaz/autoBOTLib/tree/master/results>`_. A working full CLI cycle (training+prediction) is given in `this script. <https://github.com/SkBlaz/autoBOTLib/tree/master/cli_example.sh>`_


.. code-block:: text
		
   ## Train an autoBOTLib classifier
   python autoBOTLib --mode learning --datafolder ./data/insults --output_folder results --classifier autoBOTLib-base --time 1 --hof_size 3 --representation_type neurosymbolic --mutation_rate 0.3 --crossover_rate 0.6

   ## Obtain predictions from a trained model
   python autoBOTLib --mode prediction --load_model ./results/autoBOTLibmodel.pickle --predict_data ./data/insults/test.tsv


This will result in a file called `test_predictions.tsv` in the `results` folder.
