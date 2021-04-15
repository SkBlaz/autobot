## A simple example showcasing the minimal usecase of autoBOTLib on an insults classification data.

import autoBOTLib
import pandas as pd

## Load example data frame
dataframe = pd.read_csv("../data/spanish/train.tsv", sep="\t")
train_sequences = dataframe['text_a'].values.tolist()
train_targets = dataframe['label'].values

autoBOTLibObj = autoBOTLib.GAlearner(train_sequences,
                               train_targets,
                               time_constraint=0.1).evolve()
