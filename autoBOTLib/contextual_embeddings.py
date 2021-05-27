### A wrapper for sentence-embeddings library

import logging
import numpy as np
from collections import defaultdict
import tqdm

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

from sentence_transformers import SentenceTransformer

class ContextualDocs:
    
    def __init__(self,
                 model = "paraphrase-xlm-r-multilingual-v1"):
        """
        Class initialization method.

        :param ndim: Number of latent dimensions
        :param model: The sentence-transformer model
        
        """

        try:
            self.model = SentenceTransformer(model)
            
        except Exception as es:
            logging.info(es)

    def fit(self, documents):
        """
        :param documents: The input set of documents.
        """
        
        pass

    def transform(self, documents):
        """
        :param documents: The input set of documents.
        """

        if not isinstance(documents, list):
            
            try: # Pandas
                documents = documents.values.tolist()
                
            except: # numpy
                documents = documents.tolist()
        try:
            sentence_embeddings = self.model.encode(documents)
            
        except Exception as es:
            print(es, "error in encoding documents", sentence_embeddings)
            
        encoded_documents = np.array(sentence_embeddings)
        self.ndim = encoded_documents.shape[1]
        return encoded_documents

    def fit_transform(self, documents, b = None):

        """
        :param documents: The input set of documents.
        """
        return self.transform(documents)

    def get_feature_names(self):

        """
        :param fnames: Feature names (custom api artefact)
        """
        
        return [f"dim_{x}" for x in range(self.ndim)]

if __name__ == "__main__":
    
    import pandas as pd
    
    example_text = pd.read_csv("../data/insults/train.tsv", sep="\t")['text_a']
    labels = pd.read_csv("../data/insults/train.tsv",
                         sep="\t")['label'].values.tolist()
    clx = ContextualDocs()
    sim_features = clx.fit_transform(example_text)

    print(sim_features.shape)
