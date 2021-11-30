## A simple example showcasing the minimal usecase of autoBOTLib on an insults classification data.

import autoBOTLib
import pandas as pd


def run():
    ## Load example data frame
    dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t").iloc[:200]
    train_sequences = dataframe['text_a']
    train_targets = dataframe['label']

    autoBOTLibObj = autoBOTLib.GAlearner(
        train_sequences,
        train_targets,
        representation_type=
        "neurosymbolic-lite",  ## See the documentation for all possible representation types.
        n_fold_cv=3,
        memory_storage="memory",
        sparsity=0.1,
        visualize_progress=True, ## Stores progress as PROGRESS_{generation}.pdf file
        upsample=
        False,  ## Suitable for imbalanced data - randomized upsampling tends to help.
        time_constraint=0.2).evolve(
            strategy="evolution"
        )  ## strategy = "direct-learning" trains a single learner.

    # Store
    autoBOTLib.store_autobot_model(autoBOTLibObj,
                                      f"model.pickle")

    # Load
    autoBOTObj = autoBOTLib.load_autobot_model(
        f"model.pickle")

    # Predict
    dataframe2 = pd.read_csv("../data/insults/test.tsv", sep="\t")
    test_sequences = dataframe2['text_a']
    predictions = autoBOTLibObj.predict(test_sequences)
    prob_predictions = autoBOTLibObj.predict_proba(test_sequences)
    print(predictions)
    print(prob_predictions)

    autoBOTLibObj.generate_report(output_folder="./report/",
                                  job_id="REPORTNEW")


if __name__ == "__main__":
    run()
