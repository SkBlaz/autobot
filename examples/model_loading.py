## A simple example showcasing the minimal usecase of autoBOTLib on an insults classification data.

import autoBOTLib
import pandas as pd

## Load example data frame
dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t")
train_sequences = dataframe['text_a'].values.tolist()
train_targets = dataframe['label'].values

## Simply load the model
autoBOTLibObj = autoBOTLib.load_autobot_model(
    "../stored_models/example_insults_model.pickle")
dataframe2 = pd.read_csv("../data/insults/test.tsv", sep="\t")
test_sequences = dataframe2['text_a'].values.tolist()
test_targets = dataframe2['label'].values

## Predict with the model
predictions = autoBOTLibObj.predict(test_sequences)
performance = autoBOTLib.compute_metrics(
    "first_run_task_name", predictions,
    test_targets)  ## compute F1, acc and F1_acc (as in GLUE)

## visualize performance
print(performance)
