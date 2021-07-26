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
        representation_type = "symbolic", ## See the documentation for all possible representation types.
        n_fold_cv=3,
        sparsity = 0.05,
        compress_sparse_representation = True,
        upsample = False, ## Suitable for imbalanced data - randomized upsampling tends to help.
        time_constraint=0.2).evolve(strategy = "evolution") ## strategy = "direct-learning" trains a single learner.

    dataframe2 = pd.read_csv("../data/insults/test.tsv", sep="\t")
    test_sequences = dataframe2['text_a']
    predictions = autoBOTLibObj.predict(test_sequences)
    prob_predictions = autoBOTLibObj.predict_proba(test_sequences)
    print(predictions)
    print(prob_predictions)

    autoBOTLibObj.generate_report(output_folder = "./report/")
    
if __name__ == "__main__":
    run()
