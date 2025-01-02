### A wrapper for sentence-embeddings library

import logging
import numpy as np
import tqdm
from PIL import Image

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


class ImageEmbeddingTransformer:
    def __init__(self, model="clip-ViT-B-32"):
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

            sentence_embeddings = []
            for document in tqdm.tqdm(documents):
                if isinstance(document, str):
                    loaded_image = Image.open(document)
                    
                else:
                    loaded_image = document
                sentence_embeddings.append(self.model.encode(loaded_image, show_progress_bar=False))

        except Exception as es:
            print(es, "error in encoding documents", sentence_embeddings)

        encoded_documents = np.array(sentence_embeddings).reshape(len(documents), -1)
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
    from PIL import Image


    example_text = [Image.open("../../data/multimodal_examples/tasty-recipes/raw/hummus/white-bean-cranberry-hummus.jpg")] * 10
    example_text = pd.DataFrame({"image_a": example_text})
    clx = ImageEmbeddingTransformer()
    sim_features = clx.fit_transform(example_text)

    print(sim_features.shape)
    print(sim_features)
