autoBOTLib CLI
===============
To streamline the experiments, it makes a lot of sense to directly use the *autoBOTLib* as a CLI tool. The library itself implements wrappers for main functions, and can be executed as follows. If you installed the package, the `autobot-cli` tool was also built as part of the installation. By running

.. code-block:: text

	python autobot-cli --help

The arguments are defined as follows.

.. code-block:: text
				
	usage: autobot-cli [-h] [--time TIME] [--job_id JOB_ID] [--popsize POPSIZE]
                   [--output_folder OUTPUT_FOLDER]
                   [--learner_preset LEARNER_PRESET] [--hof_size HOF_SIZE]
                   [--representation_type REPRESENTATION_TYPE]
                   [--train_data TRAIN_DATA] [--mutation_rate MUTATION_RATE]
                   [--crossover_rate CROSSOVER_RATE]
                   [--predict_data PREDICT_DATA] [--load_model LOAD_MODEL]
                   [--num_cpu NUM_CPU] [--upsample UPSAMPLE] [--mode MODE]

	optional arguments:
	  -h, --help            show this help message and exit
	  --time TIME
	  --job_id JOB_ID
	  --popsize POPSIZE
	  --output_folder OUTPUT_FOLDER
	  --learner_preset LEARNER_PRESET
	  --hof_size HOF_SIZE
	  --representation_type REPRESENTATION_TYPE
	  --train_data TRAIN_DATA
	  --mutation_rate MUTATION_RATE
	  --crossover_rate CROSSOVER_RATE
	  --predict_data PREDICT_DATA
	  --load_model LOAD_MODEL
	  --num_cpu NUM_CPU
	  --upsample UPSAMPLE
	  --mode MODE


Hence, as a minimal example, we can consider running

.. code-block:: text
				
	autobot-cli --train_data ./data/insults/train.tsv --output_folder CLI --learner_preset mini-l1

Here, the `train.tsv` file needs to have two attributes; `text_a` - the documents field, and `label`, the label field. Once the run finishes, you will have a trained model with a report in the `CLI` folder. To make predictions on unseen data, simply

.. code-block:: text
				
	autobot-cli --mode prediction --predict_data ./data/insults/test.tsv --load_model ./CLI/autoBOTmodel.pickle --output_folder CLI


See `https://github.com/skblaz/autobot/cli_example.sh` for a full example.
