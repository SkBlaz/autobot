## A simple example showcasing the minimal usecase of autoBOTLib on an insults classification data.

import autoBOTLib
import pandas as pd

def run():
    ## Load example data frame
    dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t")
    train_sequences = dataframe['text_a'].values.tolist()
    train_targets = dataframe['label'].values

    autoBOTLibObj = autoBOTLib.GAlearner(train_sequences,
                                         train_targets,
                                         n_fold_cv = 3,
                                         time_constraint=0.1).evolve()

    dataframe2 = pd.read_csv("../data/insults/test.tsv", sep="\t")
    test_sequences = dataframe2['text_a'].values.tolist()
    predictions = autoBOTLibObj.predict(test_sequences)

if __name__ == "__main__":
    run()