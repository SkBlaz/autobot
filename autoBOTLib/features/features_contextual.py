### A wrapper for sentence-embeddings library

import logging
import nltk
import numpy as np
import tqdm

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

global contextual_feature_library
try:
    from sentence_transformers import SentenceTransformer
    contextual_feature_library = True

except:
    logging.info(
        "IMPORTANT: No contextual representations will be considered. Please pip install sentence-transformers for full functionality! (this setting performs the best)"
    )
    contextual_feature_library = False


class ContextualDocs:
    def __init__(self, model="all-mpnet-base-v2"):
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

    def transform(self, documents):
        """
        :param documents: The input set of documents.
        """

        if not isinstance(documents, list):

            try:  # Pandas
                documents = documents.values.tolist()

            except:  # numpy
                documents = documents.tolist()
        try:

            # Split to sentences, embed, join
            sentence_embeddings = []
            for document in tqdm.tqdm(documents):
                sentences = nltk.sent_tokenize(document)
                doc_emb = []
                for sentence in sentences:
                    s_emb = self.model.encode(sentence, show_progress_bar=False)
                    doc_emb.append(s_emb)
                doc_emb = np.array(doc_emb)
                doc_emb = np.mean(doc_emb, axis=0)
                sentence_embeddings.append(doc_emb)

        except Exception as es:
            print(es, "error in encoding documents", sentence_embeddings)

        encoded_documents = np.array(sentence_embeddings)
        self.ndim = encoded_documents.shape[1]
        return encoded_documents

    def fit_transform(self, documents, b=None):
        """
        :param documents: The input set of documents.
        """
        return self.transform(documents)

    def get_feature_names_out(self):
        """
        :param fnames: Feature names (custom api artefact)
        """

        return [f"dim_{x}" for x in range(self.ndim)]


if __name__ == "__main__":

    import pandas as pd

    example_text = pd.read_csv("../data/dontpatronize/train.tsv",
                               sep="\t")['text_a']
    labels = pd.read_csv("../data/dontpatronize/train.tsv",
                         sep="\t")['label'].values.tolist()
    clx = ContextualDocs()
    sim_features = clx.fit_transform(example_text)

    print(sim_features.shape)
