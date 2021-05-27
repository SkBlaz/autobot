### A wrapper for sentence-embeddings library

import logging
from collections import defaultdict
import tqdm

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

from sentence_transformers import SentenceTransformer

class ContextualDocs:
    
    def __init__(self,
                 ndim = 128,
                 model = "paraphrase-xlm-r-multilingual-v1"):
        """
        Class initialization method.

        :param ndim: Number of latent dimensions
        :param model: The sentence-transformer model
        
        """

        self.ndim = ndim
        self.model = SentenceTransformer(model)

    def fit(self, documents):
        """
        :param documents: The input set of documents.
        """
        
        pass

    def transform(self, documents):
        """
        :param documents: The input set of documents.
        """
        
        sentence_embeddings = []

        ## Encode the document space
        for x in tqdm.tqdm(documents, total = len(documents)):
            sentence_embeddings.append(self.model.encode(x))
            
        encoded_documents = np.array(sentence_embeddings)

    def fit_transform(self, documents):

        """
        :param documents: The input set of documents.
        """

        return self.transform(documents)

    def get_feature_names(self, fnames):

        """
        :param fnames: Feature names (custom api artefact)
        """
        
        return [f"dim_{x}" for x in range(self.ndim)]

if __name__ == "__main__":

    example_text = pd.read_csv("../data/insults/train.tsv", sep="\t")['text_a']
    labels = pd.read_csv("../data/insults/train.tsv",
                         sep="\t")['label'].values.tolist()
    clx = ContextualDocs(ndim=512)
    sim_features = clx.fit_transform(example_text)

    print(sim_features)
