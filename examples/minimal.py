## A simple example showcasing the minimal usecase of autoBOTLib on an insults classification data.

import autoBOTLib
import pandas as pd

def run():
    ## Load example data frame
    dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t")
    train_sequences = dataframe['text_a']
    train_targets = dataframe['label']

    autoBOTLibObj = autoBOTLib.GAlearner(
        train_sequences,
        train_targets,
        n_fold_cv=5,
        time_constraint=0.05).evolve(strategy = "evolution") ## strategy = "direct-learning" trains a single learner.

    dataframe2 = pd.read_csv("../data/insults/test.tsv", sep="\t")
    test_sequences = dataframe2['text_a']
    predictions = autoBOTLibObj.predict(test_sequences)
    prob_predictions = autoBOTLibObj.predict_proba(test_sequences)
    print(predictions)
    print(prob_predictions)

    importances_local, importances_global = autoBOTLibObj.feature_type_importances(
    )
    print(importances_global)
    print(importances_local)
    importances_local.to_csv("local_insults.tsv", sep="\t")

    topic_df = autoBOTLibObj.get_topic_explanation()
    print(topic_df)
    
if __name__ == "__main__":
    run()
