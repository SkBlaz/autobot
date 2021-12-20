"""
AutoBOT. Skrlj et al. 2021
"""

from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn import pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from autoBOTLib.features.features_topic import *
from autoBOTLib.features.features_document_graph import *
from autoBOTLib.features.features_concepts import *
from autoBOTLib.features.features_keyword import *
from autoBOTLib.features.features_sentence_embeddings import *
from autoBOTLib.features.features_token_relations import *
from autoBOTLib.features.features_contextual import *

import string
import re
import multiprocessing as mp
from scipy.sparse import hstack

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import word_tokenize

import pandas as pd
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger(__name__).setLevel(logging.INFO)

try:
    import nltk
    nltk.data.path.append("nltk_data")

except Exception as es:
    import nltk
    print(es)

try:
    from nltk.tag import PerceptronTagger
except:

    def PerceptronTagger():
        return 0

global feature_presets
feature_presets = {}

# Full stack
feature_presets['neurosymbolic'] = [
    'concept_features', 'document_graph', 'neural_features_dbow',
    'neural_features_dm', 'relational_features_token', 'topic_features',
    'keyword_features', 'relational_features_char', 'char_features',
    'word_features', 'relational_features_bigram', 'contextual_features'
]

# This one is ~language agnostic
feature_presets['neurosymbolic-lite'] = [
    'document_graph', 'neural_features_dbow', 'neural_features_dm',
    'topic_features', 'keyword_features', 'relational_features_char',
    'relational_features_token', 'char_features', 'word_features',
    'relational_features_bigram', 'concept_features'
]

# MLJ paper versions
feature_presets['neurosymbolic-default'] = [
    'neural_features_dbow', 'neural_features_dm', 'keyword_features',
    'relational_features_char', 'char_features', 'word_features',
    "pos_features", 'concept_features'
]

feature_presets['neural'] = [
    'document_graph', 'neural_features_dbow', 'neural_features_dm'
]

feature_presets['symbolic'] = [
    'concept_features', 'relational_features_token', 'topic_features',
    'keyword_features', 'relational_features_char', 'char_features',
    'word_features', 'pos_features', 'relational_features_bigram'
]

if not contextual_feature_library:
    feature_presets['neurosymbolic'].pop()


def remove_punctuation(text):
    """
    This method removes punctuation
    """

    table = text.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    return text


def remove_stopwords(text):
    """
    This method removes stopwords

    :param list/pd.Series text: Input string of text
    :return str string: Preprocessed text
    """

    stops = set(stopwords.words("english"))
    text = text.split()
    text = [x.lower() for x in text if x.lower() not in stops]
    return " ".join(text)


def remove_mentions(text, replace_token):
    """
    This method removes mentions (relevant for tweets)

    :param str text: Input string of text
    :param str replace_token: A token to be replaced
    :return str string: A new text string
    """

    return re.sub(r'(?:@[\w_]+)', replace_token, text)


def remove_hashtags(text, replace_token):
    """
    This method removes hashtags

    :param str text: Input string of text
    :param str replace_token: The token to be replaced
    :return str string: A new text
    """

    return re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", replace_token, text)


def remove_url(text, replace_token):
    """
    Removal of URLs

    :param str text: Input string of text
    :param str replace_token: The token to be replaced
    :return str string: A new text
    """

    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(regex, replace_token, text)


def get_affix(text):
    """
    This method gets the affix information

    :param str text: Input text.
    """

    return " ".join(
        [word[-4:] if len(word) >= 4 else word for word in text.split()])


def get_pos_tags(text):
    """
    This method yields pos tags

    :param str text: Input string of text
    :return str string: space delimited pos tags.
    """

    tokens = word_tokenize(text)
    tgx = " ".join([x[1] for x in pos_tag(tokens)])
    return tgx


def ttr(text):
    """
    Number of unique tokens

    :param str text: Input string of text
    :return float floatValue: Ratio of the unique/overall tokens
    """

    if len(text.split(" ")) > 1 and len(text.split()) > 0:
        return len(set(text.split())) / len(text.split())
    else:
        return 0


class text_col(BaseEstimator, TransformerMixin):
    """
    A helper processor class

    :param obj BaseExtimator: Core estimator
    :param obj TransformerMixin: Transformer object
    :return obj object: Returns particular text column
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


# fit and transform numeric features, used in scikit Feature union
class digit_col(BaseEstimator, TransformerMixin):
    """
    Dealing with numeric features

    :param obj BaseExtimator: Core estimator
    :param obj TransformerMixin: Transformer object
    :return obj object: Returns transformed (scaled) space
    """
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        d_col_drops = [
            'text', 'no_punctuation', 'no_stopwords', 'text_clean', 'affixes',
            'pos_tag_seq'
        ]
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        scaler = preprocessing.MinMaxScaler().fit(hd_searches)
        return scaler.transform(hd_searches)


def parallelize(data, method):
    """
    Helper method for parallelization

    :param pd.DataFrame data: Input data to be transformed
    :param obj method: The method to parallelize
    :return pd.DataFrame data: Returns the transformed data
    """

    cores = mp.cpu_count()
    data_split = np.array_split(data, cores)
    pool = mp.Pool(cores)
    data = pd.concat(pool.map(method, data_split))
    pool.close()
    pool.join()
    return data


def build_dataframe(data_docs):
    """
    One of the core methods responsible for construction of a dataframe object.

    :param list/pd.Series data_docs: The input data documents
    :return pd.DataFrame df_data: A dataframe corresponding to text representations
    """

    df_data = pd.DataFrame({'text': data_docs})
    df_data['no_punctuation'] = df_data['text'].map(
        lambda x: remove_punctuation(x))
    df_data['no_stopwords'] = df_data['no_punctuation'].map(
        lambda x: remove_stopwords(x))
    df_data['text_clean'] = df_data['text']
    df_data['pos_tag_seq'] = df_data['text_clean'].map(
        lambda x: get_pos_tags(x))
    return df_data


class FeaturePrunner:
    """
    Core class describing sentence embedding methodology employed here.
    """
    def __init__(self, max_num_feat=2048):

        self.max_num_feat = max_num_feat

    def fit(self, input_data, y=None):

        return self

    def transform(self, input_data):
        print(input_data.shape)
        return input_data

    def get_feature_names(self):

        pass


def fast_screening_sgd(training, targets):

    parameters = {"loss": ["log"]}
    svc = SGDClassifier()
    if len(np.unique(targets)) > 2:
        f1_scoring = "f1_macro"
    else:
        f1_scoring = "f1"
        ncores = 8
        clf = GridSearchCV(svc,
                           parameters,
                           verbose=0,
                           n_jobs=ncores,
                           cv=5,
                           scoring=f1_scoring,
                           refit=False)
        clf.fit(training, targets)
        f1_perf = max(clf.cv_results_['mean_test_score'])
    return f1_perf


def get_subset(indice_list, data_matrix, vectorizer):
    current_fnum = 0
    feature_subspaces = []
    for num_feat, transformer in zip(
            indice_list, vectorizer.named_steps['union'].transformer_list):
        features = transformer[1].steps[1][1].get_feature_names()
        if num_feat <= len(features):
            subset = data_matrix[:,
                                 current_fnum:(current_fnum +
                                               min(num_feat, len(features)))]
            feature_subspaces.append(subset)
            current_fnum += len(features)
    return hstack(feature_subspaces)


def get_simple_features(df_data, max_num_feat=10000):

    tfidf_word_unigram = TfidfVectorizer(ngram_range=(1, 3),
                                         sublinear_tf=False,
                                         max_features=max_num_feat)
    tfidf_pos_unigram = TfidfVectorizer(ngram_range=(1, 3),
                                        max_features=max_num_feat)
    tfidf_char_bigram = TfidfVectorizer(analyzer='char',
                                        ngram_range=(2, 4),
                                        max_features=max_num_feat)
    lr_rel_features_unigram = relationExtractor(max_features=max_num_feat,
                                                min_token="unigrams")
    symbolic_features = [
        ('word_features',
         pipeline.Pipeline([('s1', text_col(key='no_stopwords')),
                            ('word_tfidf_unigram', tfidf_word_unigram)])),
        ('char_features',
         pipeline.Pipeline([('s2', text_col(key='no_stopwords')),
                            ('char_tfidf_bigram', tfidf_char_bigram)])),
        ('pos_features',
         pipeline.Pipeline([('s3', text_col(key='pos_tag_seq')),
                            ('pos_tfidf_unigram', tfidf_pos_unigram)]))
    ]

    feature_names = [x[0] for x in symbolic_features]
    matrix = pipeline.Pipeline([
        ('union', FeatureUnion(transformer_list=symbolic_features, n_jobs=8)),
        ('normalize', Normalizer())
    ])

    try:
        data_matrix = matrix.fit_transform(df_data)
        tokenizer = matrix

    except Exception as es:
        print(es, "Feature construction error.")
        tokenizer = None

    return tokenizer, feature_names, data_matrix


def get_features(df_data,
                 representation_type="neurosymbolic",
                 targets=None,
                 sparsity=0.1,
                 embedding_dim=512,
                 memory_location="memory",
                 custom_pipeline=None,
                 random_seed=54324,
                 normalization_norm="l2",
                 contextual_model="all-mpnet-base-v2",
                 combine_with_existing_representation=False):
    """
    Method that computes various TF-IDF-alike features.

    :param list/pd.Series df_data: The input collection of texts
    :param str representation_type: Type of representation to be used.
    :param list/np.array targets: The target space (optional)
    :param float sparsity: The hyperparameter determining the dimensionalities of separate subspaces
    :param str normalization_norm: The normalization of each subspace
    :param int embedding_dim: The latent dimension for doc. embeddings
    :param str memory_location: Location of the gzipped ConceptNet-like memory.
    :param obj custom_pipeline: Custom pipeline to be used for features if needed.
    :param str contextual_model: The language model string compatible with sentence-transformers library (this is in beta)
    :param int random_seed: The seed for the pseudo-random parts.
    :param bool combine_with_existing_representation: Whether to use existing representations + user-specified ones.
    :return obj/list/matrix: Transformer pipeline, feature names and the feature matrix.
    """

    # Seeds and np for np
    np.random.seed(random_seed)

    if not custom_pipeline is None and combine_with_existing_representation == False:

        features = custom_pipeline

    else:

        max_num_feat = int(embedding_dim / sparsity)
        logging.info(
            "Considering {} features per type, assuming sparsity of {}.".
            format(max_num_feat, sparsity))

        max_tokenizer, max_feature_names, max_data_matrix = None, None, None

        # Initialize transformer objects
        sentence_embedder_dm1 = documentEmbedder(max_features=max_num_feat,
                                                 dm=1,
                                                 ndim=embedding_dim)

        sentence_embedder_dm2 = documentEmbedder(max_features=max_num_feat,
                                                 dm=0,
                                                 ndim=embedding_dim)

        doc_sim_features = RelationalDocs(ndim=embedding_dim, targets=None)

        tfidf_word_unigram = TfidfVectorizer(ngram_range=(1, 3),
                                             sublinear_tf=False,
                                             max_features=max_num_feat)

        tfidf_pos_unigram = TfidfVectorizer(ngram_range=(1, 3),
                                            max_features=max_num_feat)

        tfidf_char_bigram = TfidfVectorizer(analyzer='char',
                                            ngram_range=(2, 4),
                                            max_features=max_num_feat)

        lr_rel_features_unigram = relationExtractor(max_features=max_num_feat,
                                                    min_token="unigrams")

        lr_rel_features_bigram = relationExtractor(max_features=max_num_feat,
                                                   min_token="bigrams")

        lr_rel_features_token = relationExtractor(max_features=max_num_feat,
                                                  min_token="word")

        keyword_features = KeywordFeatures(max_features=max_num_feat,
                                           targets=targets)

        topic_features = TopicDocs(ndim=embedding_dim)

        concept_features_transformer = ConceptFeatures(
            max_features=max_num_feat, knowledge_graph=memory_location)
        
        contextual_features = None

        if contextual_feature_library:

            if representation_type == "neurosymbolic":
                contextual_features = ContextualDocs(model=contextual_model)

            elif isinstance(representation_type,
                            list) and "neurosymbolic" in representation_type:
                contextual_features = ContextualDocs(model=contextual_model)

        feature_transformer_vault = {
            "pos_features":
            ('pos_features',
             pipeline.Pipeline([('s3', text_col(key='pos_tag_seq')),
                                ('pos_tfidf_unigram', tfidf_pos_unigram),
                                ('normalize', Normalizer(norm=normalization_norm))])),
            "word_features":
            ('word_features',
             pipeline.Pipeline([('s1', text_col(key='no_stopwords')),
                                ('word_tfidf_unigram', tfidf_word_unigram),
                                ('normalize', Normalizer(norm=normalization_norm))])),
            "char_features": ('char_features',
                              pipeline.Pipeline([
                                  ('s2', text_col(key='no_stopwords')),
                                  ('char_tfidf_bigram', tfidf_char_bigram),
                                  ('normalize', Normalizer(norm=normalization_norm))
                              ])),
            "relational_features_char":
            ('relational_features_char',
             pipeline.Pipeline([('s4', text_col(key='no_stopwords')),
                                ('relational_features_unigram',
                                 lr_rel_features_unigram),
                                ('normalize', Normalizer(norm=normalization_norm))])),
            "relational_features_bigram":
            ('relational_features_bigram',
             pipeline.Pipeline([('s10', text_col(key='no_stopwords')),
                                ('relational_features_bigram',
                                 lr_rel_features_bigram),
                                ('normalize', Normalizer(norm=normalization_norm))])),
            "keyword_features": ('keyword_features',
                                 pipeline.Pipeline([
                                     ('s5', text_col(key='no_stopwords')),
                                     ('keyword_features', keyword_features),
                                     ('normalize', Normalizer(norm=normalization_norm))
                                 ])),
            "topic_features": ('topic_features',
                               pipeline.Pipeline([
                                   ('s6', text_col(key='no_stopwords')),
                                   ('topic_features', topic_features),
                                   ('normalize', Normalizer(norm=normalization_norm))
                               ])),
            "relational_features_token":
            ('relational_features_token',
             pipeline.Pipeline([('s4', text_col(key='no_stopwords')),
                                ('relational_features_token',
                                 lr_rel_features_token),
                                ('normalize', Normalizer(norm=normalization_norm))])),
            "neural_features_dm": ('neural_features_dm',
                                   pipeline.Pipeline([
                                       ('s7', text_col(key='no_stopwords')),
                                       ('sentence_embedding_mean',
                                        sentence_embedder_dm1),
                                       ('normalize', Normalizer(norm=normalization_norm))
                                   ])),
            "neural_features_dbow": ('neural_features_dbow',
                                     pipeline.Pipeline([
                                         ('s8', text_col(key='no_stopwords')),
                                         ('sentence_embedding_mean',
                                          sentence_embedder_dm2),
                                         ('normalize', Normalizer(norm=normalization_norm))
                                     ])),
            "document_graph":
            ('document_graph',
             pipeline.Pipeline([('s9', text_col(key='no_stopwords')),
                                ('doc_similarity_features', doc_sim_features),
                                ('normalize', Normalizer(norm=normalization_norm))])),
            "concept_features": ('concept_features',
                                 pipeline.Pipeline([
                                     ('s6', text_col(key='text')),
                                     ('concept_features',
                                      concept_features_transformer),
                                     ('normalize', Normalizer(norm=normalization_norm))
                                 ])),
            "contextual_features":
            ('contextual_features',
             pipeline.Pipeline([('s6', text_col(key='text')),
                                ('contextual_features', contextual_features),
                                ('normalize', Normalizer(norm=normalization_norm))]))
        }


        if isinstance(representation_type, str):

            ## representation_type is pre-set

            if not representation_type in feature_presets:
                logging.info(
                    "Please, specify a valid preset! (see the documentation for the up-to-date namings)"
                )

            preset_features = feature_presets[representation_type]
            features = [feature_transformer_vault[x] for x in preset_features]

        else:

            # representation_type is a list of desired feature types.
            features = [
                feature_transformer_vault[x] for x in representation_type
            ]

    if not custom_pipeline is None and combine_with_existing_representation:
        features = features + custom_pipeline

    feature_names = [x[0] for x in features]
    matrix = pipeline.Pipeline([('union',
                                 FeatureUnion(transformer_list=features,
                                              n_jobs=1)),
                                ('normalize', Normalizer())], verbose=True)

    try:
        data_matrix = matrix.fit_transform(df_data)
        tokenizer = matrix

    except Exception as es:
        print(es, "Feature construction error.")
        tokenizer = None

    return tokenizer, feature_names, data_matrix
