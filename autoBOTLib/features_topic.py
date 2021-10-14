import logging
from collections import defaultdict
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import tqdm


class TopicDocs:
    
    def __init__(self,
                 ndim=128,
                 random_seed=1965123,
                 topic_tokens=8196,
                 verbose=True):
        """
        Class initialization method.

        :param ndim: Number of latent dimensions
        :param targets: The target vector
        :param random_seed: The random seed used
        :param ed_cutoff: Cutoff for fuzzy string matching when comparing documents
        :param doc_limit: The max number of documents to be considered.
        :param verbose: Whether to have the printouts
        
        """

        self.ndim = int(np.sqrt(ndim))
        self.verbose = verbose
        self.random_seed = random_seed
        self.topic_tokens = topic_tokens

        
    def fit(self, text_list):
        """
        The fit method.

        :param text_list: List of input texts
        
        """

        if not type(text_list) == list:
            text_list = text_list.values.tolist()
        self.clx = TfidfVectorizer(max_features=self.topic_tokens)
        docspace = self.clx.fit_transform(text_list).T
        fnames = [(x, y) for x, y in self.clx.vocabulary_.items()]
        fnames = [x[0] for x in sorted(fnames, key=lambda x: x[1])]
        self.clustering_algo = MiniBatchKMeans(n_clusters=self.ndim)
        clusters = self.clustering_algo.fit(docspace)
        assert len(clusters.labels_) == docspace.shape[0]
        cluster_assignments = clusters.labels_
        assert len(clusters.labels_) == len(fnames)
        self.topic_features = defaultdict(set)
        for k, v in zip(fnames, cluster_assignments):
            self.topic_features[v].add(k)

            
    def transform(self, new_documents):
        """
        Transform method.

        :param new_documents: The new set of documents to be transformed.
        :return all_embeddings: The final embedding matrix
        
        """

        if not type(new_documents) == list:
            new_documents.values.tolist()

        if self.verbose:
            logging.info("Transforming new documents.")

        new_features = np.zeros((len(new_documents), self.ndim))
        for enx, doc in tqdm.tqdm(enumerate(new_documents),
                                  total=len(new_documents)):
            parts = set(doc.lower().strip().split())
            for k, v in self.topic_features.items():
                denominator = len(v)
                overlap = len(parts.intersection(v)) / denominator
                if not overlap is None:
                    new_features[enx, k] = overlap

        return new_features

    
    def fit_transform(self, documents, b=None):
        """
        The sklearn-like fit-transform method.

        """

        self.fit(documents)
        return self.transform(documents)

    
    def get_feature_names(self):
        """
        Get feature names.
        """

        return list(["topic_" + str(x) for x in range(self.ndim)])


if __name__ == "__main__":

    example_text = pd.read_csv("../data/insults/train.tsv", sep="\t")['text_a']
    labels = pd.read_csv("../data/insults/train.tsv",
                         sep="\t")['label'].values.tolist()
    clx = TopicDocs(ndim=512)
    sim_features = clx.fit_transform(example_text)

    print(clx.get_feature_names())
    print(clx.topic_features)
