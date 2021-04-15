Using custom feature transformers
===============
You can use any custom feature transformer classes as part of the evolution. In the following examples, we demonstrate one of the most powerful functionalities of autoBOTLib - its modularity.

Using custom features
---------

Let's explore the following example.

.. code-block:: python



    ## how to use custom features?
    import autoBOTLib
    import pandas as pd

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import pipeline ## A necessary import

    ## Load example data frame
    dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t")
    train_sequences = dataframe['text_a'].values.tolist()
    train_targets = dataframe['label'].values


    ## Let's say we wish to use only the following two text-to-feature transformer objects
    tfidf_word_unigram = TfidfVectorizer(ngram_range=(1,2),
					 sublinear_tf=False,
					 max_features=100)

    tfidf_char_bigram = TfidfVectorizer(analyzer='char',
					ngram_range=(1,2),
					max_features=100)

    ## Note: You can use any transformer class that is implemented in accordance with the scikit-learn API (.fit, .transform, .fit_transform, .get_feature_names, etc.)


    ## Next, put them into a list. Note the use of text_col class.
    custom_features = [
		    ('word_features',
		     pipeline.Pipeline([('s1', autoBOTLib.feature_constructors.text_col(key='no_stopwords')),
					('word_tfidf_unigram', tfidf_word_unigram)])),
		    ('char_features',
		     pipeline.Pipeline([('s2', autoBOTLib.feature_constructors.text_col(key='no_stopwords')),
					('char_tfidf_bigram', tfidf_char_bigram)]))
		]

    ## Finally, specify this list as 
    autoBOTLibObj = autoBOTLib.GAlearner(train_sequences,
				   train_targets,
				   time_constraint = 1,
				   custom_transformer_pipeline = custom_features).evolve()


    ## Down-stream task
    dataframe2 = pd.read_csv("../data/insults/test.tsv", sep="\t")
    test_sequences = dataframe2['text_a'].values.tolist()
    predictions = autoBOTLibObj.predict(test_sequences)


Note that the only constraints for you to include a custom transformation class are the following:

1.  The class must have the sklearn-like API
2.  Join it as a series of `pipeline` objects.

To see how to implement an example custom class, you can inspect for example the
`keyword-based features <https://github.com/SkBlaz/autobot/autoBOTLib/keyword_features.py>`_.


Extending existing feature spaces
---------

Let's finally inspect how to *extend* the existing feature space of autoBOTLib with some custom features. Why would this be useful? Assume you have your one interesting feature constructor, but would like to exploit the existing ones (instead of implementing them from scratch).

.. code-block:: python
		
    import autoBOTLib
    import pandas as pd

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import pipeline  ## A necessary import

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
	     ('s1', autoBOTLib.feature_constructors.text_col(key='no_stopwords')),
	     ('word_tfidf_unigram', tfidf_word_unigram)
	 ])),
	('char_features_cusom',
	 pipeline.Pipeline([
	     ('s2', autoBOTLib.feature_constructors.text_col(key='no_stopwords')),
	     ('char_tfidf_bigram', tfidf_char_bigram)
	 ]))
    ]

    ## Finally, use the flag "combine_with_existing_representation" to append the new transformer pipeline to an existing one (e.g., neurosymbolic). This way, you can easily extend current autoBOTLib!
    autoBOTLibObj = autoBOTLib.GAlearner(
	train_sequences,
	train_targets,
	time_constraint=1,
	representation_type = "neurosymbolic",
	custom_transformer_pipeline=custom_features,
	combine_with_existing_representation = True).evolve()

    dataframe2 = pd.read_csv("../data/insults/test.tsv", sep="\t")
    test_sequences = dataframe2['text_a'].values.tolist()
    predictions = autoBOTLibObj.predict(test_sequences)

Note how you only need to specify the `combine_with_existing_representation` flag apart from the custom pipeline, which will be appended to the existing (e.g., neurosymbolic) one.
