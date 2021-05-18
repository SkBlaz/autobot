## how to use custom features?
import autoBOTLib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import pipeline  ## A necessary import


def run():
    ## Load example data frame
    dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t")
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
    autoBOTLibLibObj = autoBOTLib.GAlearner(
        train_sequences,
        train_targets,
        time_constraint=1,
        custom_transformer_pipeline=custom_features).evolve()

    dataframe2 = pd.read_csv("../data/insults/test.tsv", sep="\t")
    test_sequences = dataframe2['text_a'].values.tolist()
    autoBOTLibLibObj.predict(test_sequences)


if __name__ == "__main__":
    run()
