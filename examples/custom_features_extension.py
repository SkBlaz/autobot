import autoBOTLib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import pipeline  ## A necessary import


def run():
    ## Load example data frame
    dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t")
    train_sequences = dataframe['text_a'].values.tolist()
    train_targets = dataframe['label'].values

    ## Define custom transformer classes as in the example above
    tfidf_word_unigram = TfidfVectorizer(ngram_range=(1, 2),
                                         sublinear_tf=False,
                                         max_features=100)

    tfidf_char_bigram = TfidfVectorizer(analyzer='char',
                                        ngram_range=(1, 2),
                                        max_features=100)
    custom_features = [
        ('word_features_custom',
         pipeline.Pipeline([
             ('s1',
              autoBOTLib.feature_constructors.text_col(key='no_stopwords')),
             ('word_tfidf_unigram', tfidf_word_unigram)
         ])),
        ('char_features_cusom',
         pipeline.Pipeline([
             ('s2',
              autoBOTLib.feature_constructors.text_col(key='no_stopwords')),
             ('char_tfidf_bigram', tfidf_char_bigram)
         ]))
    ]

    ## Finally, use the flag "combine_with_existing_representation" to append the new transformer pipeline to an existing one (e.g., neurosymbolic). This way, you can easily extend current autoBOTLib!
    autoBOTLibObj = autoBOTLib.GAlearner(
        train_sequences,
        train_targets,
        time_constraint=1,
        representation_type="neurosymbolic",
        custom_transformer_pipeline=custom_features,
        combine_with_existing_representation=True).evolve()

    dataframe2 = pd.read_csv("../data/insults/test.tsv", sep="\t")
    test_sequences = dataframe2['text_a'].values.tolist()
    autoBOTLibObj.predict(test_sequences)


if __name__ == "__main__":
    run()
