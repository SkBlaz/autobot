## Regression is also possible!


## A simple example showcasing the minimal usecase of autoBOTLib on an insults classification data.

import autoBOTLib
import pandas as pd

def run():
    ## Load example data frame
    dataframe = pd.read_csv("../data/regression/regression.csv")
    train_sequences = dataframe['text']
    train_targets = dataframe['humor_rating']

    autoBOTLibObj = autoBOTLib.GAlearner(
        train_sequences,
        train_targets,
        task = "regression",
        scoring_metric = "neg_mean_absolute_percentage_error", ## What to maximize during learning
        n_fold_cv=3,
        time_constraint=0.1).evolve(strategy = "evolution")

    predictions = autoBOTLibObj.predict(train_sequences)
    print(predictions)


if __name__ == "__main__":
    run()

