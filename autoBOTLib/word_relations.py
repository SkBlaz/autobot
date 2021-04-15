### relation extractor

import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

from itertools import combinations
import operator
import pandas as pd
import numpy as np
import tqdm
import multiprocessing as mp
import scipy.sparse as sps


class relationExtractor:
    """
    The main token relation extraction class. Works for arbitrary tokens.
    """
    def __init__(self,
                 max_features=10000,
                 split_char="|||",
                 witem_separator="&&&&",
                 num_cpu=1,
                 min_token="bigrams"):

        self.max_features = max_features
        self.vocabulary = {}
        self.split_char = split_char
        self.min_token = min_token
        self.witem_separator = witem_separator

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
            elif self.min_token == "bigrams":
                tokens = []
                sstring = instance.strip()
                for enx in range(len(sstring) - 1):
                    tokens.append("".join([sstring[enx], sstring[enx + 1]]))
            elif self.min_token == "unigrams":
                tokens = []
                sstring = instance.strip()
                for enx in range(len(sstring) - 1):
                    tokens.append(sstring[enx])
            token_dict = {token: enx for enx, token in enumerate(tokens)}
            pairs = combinations(set(tokens), 2)
            for pair in pairs:
                w1, w2, distance = self.compute_distance(pair, token_dict)
                if distance > 1:
                    encoded_witem = w1 + "--" + str(distance) + "--" + w2
                    if not encoded_witem in global_distances:
                        global_distances[encoded_witem] = 0
                    global_distances[encoded_witem] += 1
        return global_distances

    def fit(self, text_vector, b=None):
        """
        Fit the model to a text vector.

        :param text_vector: The input listr of texts.
        """

        logging.info("Extracting global {} relations".format(self.min_token))
        self.global_distances = {}
        pbar = tqdm.tqdm(total=len(text_vector))
        for enx, result in enumerate(map(self.witem_kernel, text_vector)):
            pbar.update(1)
            for k, v in result.items():
                if k not in self.global_distances:
                    self.global_distances[k] = 0
                self.global_distances[k] = v
            if enx % 5 == 0:  ## sorting is not needed always
                self.global_distances = dict(
                    sorted(self.global_distances.items(),
                           key=operator.itemgetter(1),
                           reverse=True)[0:self.max_features])
        self.wit_vec = sorted(self.global_distances.items(),
                              key=operator.itemgetter(1),
                              reverse=True)[0:self.max_features]

    def get_feature_names(self):
        """
        Return exact feature names.
        """

        return [x[0] for x in self.wit_vec]

    def transform(self, text_vector):
        """
        Transform the data into suitable form.

        :param text_vector: The input list of texts.
        """

        rs = []
        cs = []
        data = []
        wit_vec = [x[0] for x in self.wit_vec]
        for enin, local_distances in tqdm.tqdm(enumerate(
                map(self.witem_kernel, text_vector)),
                                               total=len(text_vector)):
            for enx, el in enumerate(wit_vec):
                if el in local_distances and el in self.global_distances:
                    fv = 1  #local_distances[el]/self.global_distances[el]
                    if not np.isnan(fv):
                        rs.append(enin)
                        cs.append(enx)
                        data.append(fv)

        assert len(rs) == len(cs)
        m = sps.csr_matrix((data, (rs, cs)),
                           shape=(len(text_vector), self.max_features))
        return m

    def fit_transform(self, text_vector, a2):
        """
        A classifc fit-transform method.

        :param text_vector: Input list of texts.
        """

        self.fit(text_vector)
        return self.transform(text_vector)


if __name__ == "__main__":

    example_text = pd.read_csv("../data/yelp/train.tsv", sep="\t")['text_a']
    rex = relationExtractor()
    rex.fit(example_text)
    m = rex.transform(example_text)
    print(np.count_nonzero(m.todense()) / (5625 * 10000))
    print(m.shape)
