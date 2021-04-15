"""
AutoBOT. Skrlj et al. 2021
"""

import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

try:
    import nltk
    nltk.data.path.append("nltk_data")

except Exception as es:
    import nltk
    print(es)

import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import multiprocessing as mp
from nltk import pos_tag
import re
import string
try:
    from nltk.tag import PerceptronTagger
except:

    def PerceptronTagger():
        return 0

## Feature constructors
from .word_relations import *
from .sentence_embeddings import *
from .keyword_features import *
from .conceptnet_features import *

## sklearn dependencies
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn import pipeline
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing

## Seeds and np for np
np.random.seed(456238)

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

    :param text: Input string of text
    :return string: Preprocessed text
    """

    stops = set(stopwords.words("english"))
    text = text.split()
    text = [x.lower() for x in text if x.lower() not in stops]
    return " ".join(text)


def remove_mentions(text, replace_token):
    """
    This method removes mentions (relevant for tweets)
    
    :param text: Input string of text
    :param replace_token: A token to be replaced
    :return string: A new text string
    """

    return re.sub(r'(?:@[\w_]+)', replace_token, text)


def remove_hashtags(text, replace_token):
    """
    This method removes hashtags

    :param text: Input string of text
    :param replace_token: The token to be replaced
    :return string: A new text
    """

    return re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", replace_token, text)


def remove_url(text, replace_token):
    """
    Removal of URLs

    :param text: Input string of text
    :param replace_token: The token to be replaced
    :return string: A new text
    """

    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(regex, replace_token, text)


def get_affix(text):
    """
    This method gets the affix information
    """

    return " ".join(
        [word[-4:] if len(word) >= 4 else word for word in text.split()])


def get_pos_tags(text):
    """
    This method yields pos tags

    :param text: Input string of text
    :return string: space delimited pos tags.
    """

    tokens = word_tokenize(text)
    tgx = " ".join([x[1] for x in pos_tag(tokens)])
    return tgx


def ttr(text):
    """
    Number of unique tokens
    
    :param text: Input string of text
    :return float: Ratio of the unique/overall tokens
    """
    
    if len(text.split(" ")) > 1 and len(text.split()) > 0:
        return len(set(text.split())) / len(text.split())
    else:
        return 0


class text_col(BaseEstimator, TransformerMixin):
    """
    A helper processor class

    :param BaseExtimator: Core estimator
    :param TransformerMixin: Transformer object
    :return object: Returns particular text column
    """
    
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


#fit and transform numeric features, used in scikit Feature union
class digit_col(BaseEstimator, TransformerMixin):
    """
    Dealing with numeric features
    
    :param BaseExtimator: Core estimator
    :param TransformerMixin: Transformer object
    :return object: Returns transformed (scaled) space
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

    :param data: Input data to be transformed
    :param method: The method to parallelize
    :return data: Returns the transformed data
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

    :param data_docs: The input data documents
    :return df_data: A dataframe corresponding to text representations
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
                 memory_location="memory/conceptnet.txt.gz",
                 custom_pipeline = None,
                 concept_features = True,
                 combine_with_existing_representation = False):
    """
    Method that computes various TF-IDF-alike features.
    """

    if not custom_pipeline is None and combine_with_existing_representation == False:

        features = custom_pipeline

    else:
        
        max_num_feat = int(embedding_dim / sparsity)
        logging.info(
            "Considering {} features per type, assuming sparsity of {}.".
            format(max_num_feat, sparsity))

        max_tokenizer, max_feature_names, max_data_matrix = None, None, None
        logging.info("Constructing {} features.".format(representation_type))
        if representation_type == "neural" or representation_type == "neurosymbolic":
            sentence_embedder_dm1 = documentEmbedder(max_features=max_num_feat,
                                                     dm=1,
                                                     ndim=embedding_dim)
            sentence_embedder_dm2 = documentEmbedder(max_features=max_num_feat,
                                                     dm=0,
                                                     ndim=embedding_dim)
            neural_features = [
                ('neural_features_v1',
                 pipeline.Pipeline([('s6', text_col(key='no_stopwords')),
                                    ('sentence_embedding_mean',
                                     sentence_embedder_dm1)])),
                ('neural_features_v2',
                 pipeline.Pipeline([('s7', text_col(key='no_stopwords')),
                                    ('sentence_embedding_mean',
                                     sentence_embedder_dm2)]))
            ]

        if representation_type == "symbolic" or representation_type == "neurosymbolic":
            tfidf_word_unigram = TfidfVectorizer(ngram_range=(1, 3),
                                                 sublinear_tf=False,
                                                 max_features=max_num_feat)

            tfidf_pos_unigram = TfidfVectorizer(ngram_range=(1, 3),
                                                max_features=max_num_feat)

            tfidf_char_bigram = TfidfVectorizer(analyzer='char',
                                                ngram_range=(2, 4),
                                                max_features=max_num_feat)

            lr_rel_features_unigram = relationExtractor(
                max_features=max_num_feat, min_token="unigrams")

            keyword_features = KeywordFeatures(max_features=max_num_feat,
                                               targets=targets)

            symbolic_features = [
                ('word_features',
                 pipeline.Pipeline([('s1', text_col(key='no_stopwords')),
                                    ('word_tfidf_unigram', tfidf_word_unigram)
                                    ])),
                ('char_features',
                 pipeline.Pipeline([('s2', text_col(key='no_stopwords')),
                                    ('char_tfidf_bigram', tfidf_char_bigram)
                                    ])),
                ('pos_features',
                 pipeline.Pipeline([('s3', text_col(key='pos_tag_seq')),
                                    ('pos_tfidf_unigram', tfidf_pos_unigram)
                                    ])),
                ('relational_features',
                 pipeline.Pipeline([('s4', text_col(key='no_stopwords')),
                                    ('relational_features_unigram',
                                     lr_rel_features_unigram)])),
                ('keyword_features',
                 pipeline.Pipeline([('s5', text_col(key='no_stopwords')),
                                    ('keyword_features', keyword_features)]))
            ]
            
            if concept_features:
                concept_features = ConceptFeatures(max_features=max_num_feat,
                                                   knowledge_graph=memory_location)

                cfx = ('concept_features',
                       pipeline.Pipeline([('s6', text_col(key='no_stopwords')),
                                          ('concept_features', concept_features)]))
                
                symbolic_features.append(cfx)

        if representation_type == "symbolic":
            features = symbolic_features

        elif representation_type == "neurosymbolic":
            features = symbolic_features + neural_features

        else:
            features = neural_features

    if not custom_pipeline is None and combine_with_existing_representation:
        features = features + custom_pipeline
        
    feature_names = [x[0] for x in features]
    matrix = pipeline.Pipeline([('union',
                                 FeatureUnion(transformer_list=features,
                                              n_jobs=1)),
                                ('normalize', Normalizer())])

    try:
        data_matrix = matrix.fit_transform(df_data)
        tokenizer = matrix

    except Exception as es:
        print(es, "Feature construction error.")
        tokenizer = None

    return tokenizer, feature_names, data_matrix


def get_autoBOT_manual(train_sequences,
                       dev_sequences,
                       train_targets,
                       dev_targets,
                       time_constraint=1,
                       num_cpu=1,
                       max_features=1000,
                       clf_type="LR"):

    total_sequences_training = train_sequences.values.tolist(
    ) + dev_sequences.values.tolist()
    total_sequences_training = build_dataframe(total_sequences_training)

    total_labels_training = train_targets.tolist() + dev_targets.tolist()
    max_num_feat = 10000
    max_tokenizer, max_feature_names, max_data_matrix = None, None, None
    representation_type = "neurosymbolic"
    embedding_dim = 512
    if representation_type == "neural" or representation_type == "neurosymbolic":
        sentence_embedder_dm1 = documentEmbedder(max_features=max_num_feat,
                                                 dm=1,
                                                 ndim=embedding_dim)
        sentence_embedder_dm2 = documentEmbedder(max_features=max_num_feat,
                                                 dm=0,
                                                 ndim=embedding_dim)
        neural_features = [
            ('neural_features_v1',
             pipeline.Pipeline([('s6', text_col(key='no_stopwords')),
                                ('sentence_embedding_mean',
                                 sentence_embedder_dm1)])),
            ('neural_features_v2',
             pipeline.Pipeline([('s7', text_col(key='no_stopwords')),
                                ('sentence_embedding_mean',
                                 sentence_embedder_dm2)]))
        ]

    if representation_type == "symbolic" or representation_type == "neurosymbolic":
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
        keyword_features = KeywordFeatures(max_features=max_num_feat,
                                           targets=train_targets)
        concept_features = ConceptFeatures(max_features=max_num_feat)
        symbolic_features = [
            ('word_features',
             pipeline.Pipeline([('s1', text_col(key='no_stopwords')),
                                ('word_tfidf_unigram', tfidf_word_unigram)])),
            ('char_features',
             pipeline.Pipeline([('s2', text_col(key='no_stopwords')),
                                ('char_tfidf_bigram', tfidf_char_bigram)])),
            ('pos_features',
             pipeline.Pipeline([('s3', text_col(key='pos_tag_seq')),
                                ('pos_tfidf_unigram', tfidf_pos_unigram)])),
            ('relational_features',
             pipeline.Pipeline([('s4', text_col(key='no_stopwords')),
                                ('relational_features_unigram',
                                 lr_rel_features_unigram)])),
            ('keyword_features',
             pipeline.Pipeline([('s5', text_col(key='no_stopwords')),
                                ('keyword_features', keyword_features)])),
            ('concept_features',
             pipeline.Pipeline([('s6', text_col(key='no_stopwords')),
                                ('concept_features', concept_features)]))
        ]

    features = symbolic_features + neural_features
    [x[0] for x in features]

    if clf_type == "LR":
        clf = LogisticRegression()

    elif clf_type == "SVM":
        clf = LinearSVC()

    cpipeline = pipeline.Pipeline([('union',
                                    FeatureUnion(transformer_list=features,
                                                 n_jobs=1)),
                                   ('normalize', Normalizer()),
                                   ('classifier', clf)])
    return cpipeline.fit(total_sequences_training, total_labels_training)
