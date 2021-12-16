### relation extractor

import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

from itertools import combinations
import operator
import pandas as pd
from collections import defaultdict
import numpy as np
from scipy import sparse
import tqdm
import multiprocessing as mp
import scipy.sparse as sps
from sklearn.preprocessing import LabelEncoder
from scipy.stats.stats import pearsonr


class relationExtractor:
    """
    The main token relation extraction class. Works for arbitrary tokens.
    """
    def __init__(
            self,
            max_features=10000,
            split_char="|||",
            witem_separator="&&&&",
            num_cpu=8,
            neighborhood_token=64,  ## Context window size for relation mapping of words (added in >.34)
            min_token="bigrams",
            targets=None,
            verbose=True):

        self.max_features = max_features
        self.neighborhood_token = neighborhood_token
        self.num_cpu = num_cpu
        self.vocabulary = {}
        self.verbose = verbose
        self.split_char = split_char
        self.min_token = min_token
        self.targets = targets
        self.witem_separator = witem_separator
        self.ranking_conducted = False

        if num_cpu == "all":
            self.num_cpu = mp.cpu_count()

        else:
            self.num_cpu = num_cpu

            
    def compute_distance(self, pair, token_dict):
        """
        A core distance for computing index-based differences.

        :param pair: the pair of tokens
        :param token_dict: distance map
        :return pair[0], pair[1], dist: The two tokens and the distance
        """

        i1 = token_dict[pair[0]]
        i2 = token_dict[pair[1]]
        dist = np.abs(i1 - i2)
        return pair[0], pair[1], dist

    
    def witem_kernel(self, instance):
        """
        A simple kernel for traversing a given document.

        :param instance: a piece of text
        :return global_distances: Distances between tokens
        """

        global_distances = {}

        if self.split_char in instance:
            instances = instance.split(self.split_char)

        else:
            instances = [instance]

        for instance in instances:

            if self.min_token == "word":
                tokens = [x for x in instance.strip().split()]

            elif self.min_token == "threegrams":
                tokens = []
                sstring = instance.strip()
                for enx in range(len(sstring) - 1):
                    tokens.append(sstring[enx:(enx + 3)].lower())

            elif self.min_token == "bigrams":
                tokens = []
                sstring = instance.strip()
                for enx in range(len(sstring) - 1):
                    tokens.append(sstring[enx:(enx + 2)].lower())

            elif self.min_token == "unigrams":
                tokens = []
                sstring = instance.strip()
                for enx in range(len(sstring) - 1):
                    tokens.append(sstring[enx])

            token_dict = {token: enx for enx, token in enumerate(tokens)}
            pairs = list(combinations(set(tokens), 2))

            for pair in pairs:
                w1, w2, distance = self.compute_distance(pair, token_dict)
                if self.min_token == "word" or self.min_token == "bigrams":
                    if distance > self.neighborhood_token:
                        continue

                if distance > 2:
                    context_size = int(np.log2(distance))
                    encoded_witem = w1 + "--" + str(context_size) + "--" + w2
                    if not encoded_witem in global_distances:
                        global_distances[encoded_witem] = 0
                    global_distances[encoded_witem] += 1

        return global_distances

    
    def fit(self, text_vector, b=None):
        """
        Fit the model to a text vector.

        :param text_vector: The input list of texts.
        """

        logging.info("Extracting global {} relations".format(self.min_token))
        self.global_distances = defaultdict(list)
        pbar = tqdm.tqdm(total=len(text_vector))
        for enx, result in enumerate(map(self.witem_kernel, text_vector)):
            pbar.update(1)
            for k, v in result.items():
                self.global_distances[k].append(
                    v)  ## TODO: Streaming mean. This will save some memory.

        for k, v in self.global_distances.items():
            v = np.mean(v)  # average distance

        ## This is the key part -> select the top k features within the context

        if self.targets is None:
            self.wit_vec = sorted(self.global_distances.items(),
                                  key=operator.itemgetter(1),
                                  reverse=True)[0:self.max_features]
        else:
            self.wit_vec = sorted(self.global_distances.items(),
                                  key=operator.itemgetter(1),
                                  reverse=True)[0:(self.max_features * 3)]

            feature_matrix_whole = self.transform(text_vector,
                                                  custom_shape=len(
                                                      self.wit_vec))
            if self.verbose:
                logging.info(
                    f"Feature ranking of {len(self.wit_vec)} token pairs in progress."
                )

            encoder = LabelEncoder()
            self.targets = encoder.fit_transform(self.targets)
            correlations = np.zeros(len(self.wit_vec))
            for el in tqdm.tqdm(range(len(self.wit_vec)),
                                total=len(self.wit_vec)):
                vec1 = feature_matrix_whole[:, el].todense().A1
                assert len(self.targets) == len(vec1)
                correlation = pearsonr(vec1, self.targets)[0]
                correlations[el] = correlation

            if self.verbose:
                most_correlated = np.sort(correlations)[::-1][0:10]
                logging.info("Highest correlations: {most_correlated}")

            sorted_correlations = set(
                np.argsort(correlations)[::-1][0:self.max_features].tolist())
            self.wit_vec = [
                x for enx, x in enumerate(self.wit_vec)
                if enx in sorted_correlations
            ]

            
    def get_feature_names(self):
        """
        Return exact feature names.
        """

        return [x[0] for x in self.wit_vec]

    
    def transform(self, text_vector, custom_shape=None):
        """
        Transform the data into suitable form.

        :param text_vector: The input list of texts.
        """

        if custom_shape is None:
            custom_shape = self.max_features

        rs = []
        cs = []
        data = []
        wit_vec = [x[0] for x in self.wit_vec]
        for enin, local_distances in tqdm.tqdm(enumerate(
                map(self.witem_kernel, text_vector)),
                                               total=len(text_vector)):
            for enx, el in enumerate(wit_vec):
                if el in local_distances:
                    fv = local_distances[el]
                    if not np.isnan(fv):
                        rs.append(enin)
                        cs.append(enx)
                        data.append(fv)

        assert len(rs) == len(cs)
        m = sps.csr_matrix((data, (rs, cs)),
                           shape=(len(text_vector), custom_shape))
        return m

    
    def fit_transform(self, text_vector, a2):
        """
        A classifc fit-transform method.

        :param text_vector: Input list of texts.
        """

        self.fit(text_vector)
        return self.transform(text_vector)


if __name__ == "__main__":

    tfile = pd.read_csv("../data/insults/train.tsv", sep="\t")
    example_text = tfile['text_a']
    targets = tfile['label']
    rex = relationExtractor(min_token="threegrams")
    rex.fit(example_text)
    m = rex.transform(example_text)
    print(np.count_nonzero(m.todense()) / (5625 * 10000))
    print(np.isnan(m.todense()).any())
    print(m.shape)
    print(rex.get_feature_names()[0:30])
