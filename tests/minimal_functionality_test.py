## A simple example showcasing the minimal usecase of autoBOTLib on an insults classification data.

import autoBOTLib
import pandas as pd
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import pipeline  ## A necessary import


def test_minimal():
    ## Load example data frame
    dataframe = pd.read_csv("./data/insults/train.tsv", sep="\t")
    train_sequences = dataframe['text_a'].values.tolist()
    train_targets = dataframe['label'].values

    autoBOTLibObj = autoBOTLib.GAlearner(train_sequences,
                                         train_targets,
                                         time_constraint=0.1).evolve()

    dataframe2 = pd.read_csv("./data/insults/test.tsv", sep="\t")
    test_sequences = dataframe2['text_a'].values.tolist()
    autoBOTLibObj.predict(test_sequences)


def test_custom_classifier():
    dataframe = pd.read_csv("./data/insults/train.tsv", sep="\t")
    train_sequences = dataframe['text_a'].values.tolist()
    train_targets = dataframe['label'].values

    ## The syntax for specifying a learner and the hyperparameter space!
    ## These are the hyperparameters to be explored for each representation.
    classifier_hyperparameters = {
        "loss": ["hinge"],
        "penalty": ["elasticnet"],
        "alpha": [0.01, 0.001],
        "l1_ratio": [0, 0.001, 1]
    }

    ## This is the classifier compatible with the hyperparameters.
    custom_classifier = SGDClassifier()

    autoBOTLibObj = autoBOTLib.GAlearner(
        train_sequences,  # input sequences
        train_targets,  # target space 
        time_constraint=1,  # time in hours
        num_cpu=4,  # number of CPUs to use
        task_name="example test",  # task identifier
        hof_size=3,  # size of the hall of fame
        top_k_importances=25,  # how many top features to output as final ranking
        memory_storage="../memory/conceptnet.txt.gz",
        representation_type="symbolic",
        classifier=custom_classifier,
        classifier_hyperparameters=classifier_hyperparameters
    )  # or neurosymbolic or neural

    autoBOTLibObj.evolve(
        nind=10,  ## population size
        strategy="evolution",  ## optimization strategy
        crossover_proba=0.6,  ## crossover rate
        mutpb=0.4)  ## mutation rate


def test_custom_features():

    ## Load example data frame
    dataframe = pd.read_csv("./data/insults/train.tsv", sep="\t")
    train_sequences = dataframe['text_a'].values.tolist()
    train_targets = dataframe['label'].values

    ## Let's say we wish to use only the following two text-to-feature transformer objects
    tfidf_word_unigram = TfidfVectorizer(ngram_range=(1, 2),
                                         sublinear_tf=False,
                                         max_features=100)

    tfidf_char_bigram = TfidfVectorizer(analyzer='char',
                                        ngram_range=(1, 2),
                                        max_features=100)

    ## Note: You can use any transformer class that is implemented in accordance with the scikit-learn API (.fit, .transform, .fit_transform, .get_feature_names, etc.)

    ## Next, put them into a list. Note the use of text_col class.
    custom_features = [
        ('word_features',
         pipeline.Pipeline([
             ('s1',
              autoBOTLib.feature_constructors.text_col(key='no_stopwords')),
             ('word_tfidf_unigram', tfidf_word_unigram)
         ])),
        ('char_features',
         pipeline.Pipeline([
             ('s2',
              autoBOTLib.feature_constructors.text_col(key='no_stopwords')),
             ('char_tfidf_bigram', tfidf_char_bigram)
         ]))
    ]

    ## Finally, specify this list as
    autoBOTLibObj = autoBOTLib.GAlearner(
        train_sequences,
        train_targets,
        time_constraint=1,
        custom_transformer_pipeline=custom_features).evolve()

    dataframe2 = pd.read_csv("./data/insults/test.tsv", sep="\t")
    test_sequences = dataframe2['text_a'].values.tolist()
    autoBOTLibObj.predict(test_sequences)
