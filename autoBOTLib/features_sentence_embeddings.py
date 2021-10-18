### relation extractor

import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

import pandas as pd
import string
import numpy as np
import tqdm
import multiprocessing as mp

from scipy import sparse
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess


class documentEmbedder:
    """
    Core class describing sentence embedding methodology employed here. 
    The class functions as a sklearn-like object.
    """
    def __init__(self,
                 max_features=10000,
                 num_cpu=8,
                 dm=1,
                 pretrained_path="doc2vec.bin",
                 ndim=512):
        """The standard sgn function.

        :param max_features: integer, number of latent dimensions
        :param num_cpu: integer, number of CPUs to be used
        :param dm:  Whether to use the "distributed memory" model
        :param pretrained_path: The path where a pretrained model is located (if any)
        """

        self.max_features = max_features
        self.dm = dm
        self.pretrained_path = pretrained_path
        self.vocabulary = {}
        self.ndim = ndim
        self.model = None
        if num_cpu == "all":
            self.num_cpu = mp.cpu_count()

        else:
            self.num_cpu = num_cpu

            
    def fit(self, text_vector, b=None, refit=False):
        """
        Fit the model to a text vector.
        :param text_vector: a list of texts
        """

        if self.model is None and not refit:

            documents = [
                TaggedDocument(simple_preprocess(doc), [i])
                for i, doc in enumerate(text_vector.values.tolist())
            ]
            self.model = Doc2Vec(vector_size=self.ndim,
                                 window=5,
                                 min_count=1,
                                 workers=self.num_cpu,
                                 dm=self.dm)
            self.model.build_vocab(documents)
            self.model.train(documents,
                             total_examples=self.model.corpus_count,
                             epochs=32)
            self.model.delete_temporary_training_data(
                keep_doctags_vectors=True, keep_inference=True)

            
    def transform(self, text_vector):
        """
        Transform the data into suitable form.
        :param text_vector: The text vector to be transformed via a trained model
        """

        set(string.printable)
        final_matrix = np.zeros((len(text_vector), self.ndim))
        for enx, doc in enumerate(tqdm.tqdm(text_vector)):
            if len(doc) > 1:
                try:
                    vector = self.model.infer_vector(simple_preprocess(doc))
                    final_matrix[enx] = vector
                except:
                    ## invalid inference.
                    pass

        logging.info("Generated embeddings ({}) of shape {}".format(
            self.dm, final_matrix.shape))

        return sparse.csr_matrix(final_matrix)

    
    def get_feature_names(self):

        return [str(x) + "_" + str(self.dm) for x in list(range(self.ndim))]

    
    def fit_transform(self, text_vector, a2=None):
        """
        A classifc fit-transform method.
        :param text_vector: a text vector used to build and transform a corpus.
        """

        self.fit(text_vector)
        return self.transform(text_vector)


if __name__ == "__main__":

    example_text = pd.read_csv("../data/dontpatronize/train.tsv",
                               sep="\t")['text_a']

    rex = documentEmbedder(dm=1)
    rex.fit(example_text)

    m = rex.transform(example_text)

    print("+" * 100)
    m = rex.fit_transform(example_text)
    print(m)
