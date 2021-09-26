## A simple example showcasing the minimal usecase of autoBOTLib on an insults classification data.

import autoBOTLib
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import pipeline  ## A necessary import
import pytest

# def test_minimal():
#     ## Load example data frame
#     dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t")
#     train_sequences = dataframe['text_a']
#     train_targets = dataframe['label']

#     autoBOTLibObj = autoBOTLib.GAlearner(
#         train_sequences,
#         train_targets,
#         representation_type=
#         "symbolic",  ## See the documentation for all possible representation types.
#         n_fold_cv=3,
#         memory_storage="memory2",
#         sparsity=0.1,
#         upsample=
#         False,  ## Suitable for imbalanced data - randomized upsampling tends to help.
#         time_constraint=0.2).evolve(
#             strategy="evolution"
#         )  ## strategy = "direct-learning" trains a single learner.

#     dataframe2 = pd.read_csv("../data/insults/test.tsv", sep="\t")
#     test_sequences = dataframe2['text_a']
#     predictions = autoBOTLibObj.predict(test_sequences)
#     prob_predictions = autoBOTLibObj.predict_proba(test_sequences)
#     print(predictions)
#     print(prob_predictions)

#     autoBOTLibObj.generate_report(output_folder="./report/",
#                                   job_id="as9y0gb98s")

@pytest.mark.parametrize("representation_type", ["symbolic","neurosymbolic-default"])
@pytest.mark.parametrize("fold_number", [2,3,4,5,6,7,8,9,10])
@pytest.mark.parametrize("sparsity", [0.01,0.05,0.1,0.3,0.6,0.9,1])
@pytest.mark.parametrize("time_constraint", [0.1,1,10])
def test_initializations(fold_number, representation_type, sparsity, time_constraint):

    dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t")
    train_sequences = dataframe['text_a']
    train_targets = dataframe['label']
    autoBOTLibObj = autoBOTLib.GAlearner(
        train_sequences,
        train_targets,
        representation_type=
        representation_type,  ## See the documentation for all possible representation types.
        n_fold_cv=fold_number,
        memory_storage="memory2",
        sparsity=sparsity,
        upsample=
        False,  ## Suitable for imbalanced data - randomized upsampling tends to help.
        time_constraint=time_constraint)
