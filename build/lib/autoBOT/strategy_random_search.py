"""
This file contains code on random search across tfidf parameter space. Skrlj 2019
"""

import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
from feature_constructors import Normalizer, build_dataframe, get_simple_features, pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import numpy as np
import time

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore')


def random_search():
    params = np.random.uniform(low=0.00001, high=1, size=2)
    return params


def evaluate_learner(cvar, train_features, train_targets, dev_features,
                     dev_targets):
    """
    Learner evaluation method.

    :param cvar: Regularization level.
    :param train_features: Train feature space.
    :param train_targets: Train target space.
    :param dev_features: Development feature space.
    :param dev_targets: Development target space.
    :return f1,clf: The F1 score and the classifier.
    """

    clf = LinearSVC(C=cvar).fit(train_features, train_targets)
    predictions = clf.predict(dev_features)

    try:
        f1 = f1_score(dev_targets, predictions)

    except:
        f1 = f1_score(dev_targets, predictions, average="micro")

    return f1, clf


def evolve_representation_random(train_seq,
                                 dev_seq,
                                 train_targets,
                                 dev_targets,
                                 time_constraint=1):
    """
    Random representation exploration.
    
    :param train_seq: training sequences.
    :param dev_seq: development sequences.
    :param train_targets: training targets.
    :param dev taragets: dev targets.
    :param time_constraint: time in hours (int)
    """

    copt = 0
    initial_time = time.time()
    time_diff = 0
    total_iterations = 0
    top_vectorizer = None
    top_learner = None
    mnf = 0
    while time_diff <= time_constraint:
        total_iterations += 1
        pars = random_search()
        if total_iterations % 2 == 0:
            logging.info(
                "Best: {}, running time: {}min, iterations: {}".format(
                    copt,
                    np.round(time_diff, 1) * 60, total_iterations))
        time_diff = ((time.time() - initial_time) / 60) / 60
        train_seq1 = build_dataframe(train_seq)
        dev_seq1 = build_dataframe(dev_seq)
        vectorizer, feature_names, _ = get_simple_features(train_seq1,
                                                           max_num_feat=2048)
        svm_c = pars[0]
        if vectorizer:
            train_bow = vectorizer.transform(train_seq1)
            num_features = train_bow.shape[1]
            dev_bow = vectorizer.transform(dev_seq1)
            f1_perf, clf = evaluate_learner(svm_c, train_bow, train_targets,
                                            dev_bow, dev_targets)
            f1_perf = np.round(f1_perf, 3)
            if f1_perf > copt:
                copt = f1_perf
                top_vectorizer = vectorizer
                mnf = int(pars[1] * 5000)
                top_learner = clf
                logging.info(
                    "Improved performance to {}! num features: {}, running time: {}min"
                    .format(f1_perf, num_features,
                            np.round(time_diff, 0) * 60))

    logging.info(
        "Finished optimization with best performance: {}".format(copt))
    auml_pip = pipeline.Pipeline([('union', top_vectorizer),
                                  ('scale', Normalizer()),
                                  ('classifier', top_learner)])
    logging.info("Optimization finished!")
    return (auml_pip, mnf)
