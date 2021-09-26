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
        learner_preset="mini-l1",
        validation_type=
        "train_test",  ## This parallelizes at the individual (not learner) level -> this results in additional memory overhead as shown in the paper.
        validation_percentage=0.15,
        num_cpu=10,
        representation_type=
        "neurosymbolic-lite",  ## full representation space -- note that this includes sentence-transformers. For a lightweight version, consider neurosymbolic-lite
        time_constraint=0.1).evolve(
            strategy="evolution"
        )  ## strategy = "direct-learning" trains a single learner.

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
