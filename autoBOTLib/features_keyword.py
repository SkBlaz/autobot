## generate supervised features via keywords.

## idea:
## group docs by classes
## for each class, find keywords,
## score w.r.t., class presence, sort, take topn
### relation extractor

import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

import pandas as pd
import numpy as np
import collections
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from .misc_keyword_detection import RakunDetector, defaultdict


class KeywordFeatures:
    """
    Core class describing sentence embedding methodology employed here. 
    """
    def __init__(self, max_features=10000, targets=None):

        self.max_features = max_features
        self.targets = targets

        
    def fit(self, text_vector, refit=False):
        """
        Fit the model to a text vector.

        :param text_vector: The input list of texts
        """

        logging.info("Starting keyword detection.")
        unique_classes = set(self.targets.tolist())
        class_keywords = {}
        for uc in unique_classes:
            hyperparameters = {
                "distance_threshold": 3,
                "distance_method": "editdistance",
                "num_keywords": 30,
                "pair_diff_length": 3,
                "stopwords": stopwords.words('english'),
                "bigram_count_threshold": 2,
                "max_occurrence": 5,
                "max_similar": 3,
                "num_tokens": [1, 2]
            }

            indices = np.where(self.targets == uc)[0].tolist()
            local_corpus = []
            try:
                text_vector = text_vector.values

            except:
                text_vector = text_vector

            for j in indices:
                try:
                    local_corpus.append(text_vector[j])

                except Exception as es:
                    print(es, "TPX")

            keyword_detector = RakunDetector(hyperparameters, verbose=False)

            all_keywords = []
            keyword_dict = defaultdict(list)

            for el in tqdm.tqdm(local_corpus):
                try:
                    if "|||" in el:  ## This is a special charset used to delimit multidocument instances
                        docs = el.split("|||")
                    else:
                        docs = [el]
                    docs_merged = "\n".join(docs)
                    keywords = keyword_detector.find_keywords(
                        docs_merged, input_type="text")
                    all_keywords += keywords

                except Exception as es:
                    print(es, "TPX2")

            for kw in all_keywords:
                keyword_dict[kw[0]].append(kw[1])

            new_dict = {}
            for k, v in keyword_dict.items():
                if not len(k) > 20:
                    new_dict[k] = np.mean(v)
            unique_keywords = set(
                collections.OrderedDict(sorted(new_dict.items(),
                                               reverse=True)).keys())
            class_keywords[uc] = unique_keywords

        key_docs = []
        for doc, label in zip(text_vector, self.targets):
            tokens = set(doc.split(" "))
            specific_keywords = class_keywords[label]
            intersection = " ".join(
                list(tokens.intersection(specific_keywords)))
            key_docs.append(intersection)

        logging.info("Keyword clustering obtained.")
        self.keyword_vectorizer = TfidfVectorizer(
            ngram_range=(1, 1), max_features=self.max_features).fit(key_docs)

        
    def transform(self, text_vector):
        """
        Transform the data into suitable form.

        :param text_vector: The input list of texts.
        :return transformedObject: The transformed input texts (feature space)
        """

        return self.keyword_vectorizer.transform(text_vector)

    
    def get_feature_names(self):

        return self.keyword_vectorizer.get_feature_names()

    
    def fit_transform(self, text_vector, b=None):
        """
        A classifc fit-transform method.

        :param text_vector: Input list of texts.
        :return transformedObject: Transformed list of texts
        """

        self.fit(text_vector, self.targets)
        return self.transform(text_vector)


if __name__ == "__main__":

    example_text = pd.read_csv("../data/spanish/train.tsv", sep="\t")
    text = example_text['text_a']
    labels = example_text['label']

    rex = KeywordFeatures(targets=labels)
    rex.fit(text)
    m = rex.transform(text)
    print(m)
