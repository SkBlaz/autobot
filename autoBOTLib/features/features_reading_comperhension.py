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

import os
import nltk
import math
import re
import pandas as pd
from collections import defaultdict

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def sylco(word):
    word = word.lower()

    syls = 0  # added syllable number
    disc = 0  # discarded syllable number

    # 1) if letters < 3 : return 1
    if len(word) <= 3:
        syls = 1
        return syls

    # 2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies", discard "es" and "ed" at the end.
    # if it has only 1 vowel or 1 set of consecutive vowels, discard. (like "speed", "fled" etc.)
    # 4) check if consecutive vowels exists, triplets or pairs, count them as one.

    doubleAndtripple = len(re.findall(r'[eaoui][eaoui]', word))
    tripple = len(re.findall(r'[eaoui][eaoui][eaoui]', word))
    disc += doubleAndtripple + tripple

    # 5) count remaining vowels in word.
    numVowels = len(re.findall(r'[eaoui]', word))


    # 9) if starts with "tri-" or "bi-" and is followed by a vowel, add one.
    if word[:3] == "tri" and word[3] in "aeoui":
        syls += 1

    if word[:2] == "bi" and word[2] in "aeoui":
        syls += 1

    # 10) if ends with "-ian", should be counted as two syllables, except for "-tian" and "-cian"

    # 13) check for "-n't" and cross match with dictionary to add syllable.
     # calculate the output
    return numVowels - disc + syls


def gfi(text):
    # Gunning Fog Index
    word_tokens = nltk.tokenize.regexp_tokenize(text, r'\w+')
    lengthW = len(word_tokens)

    sents = sent_tokenizer.tokenize(text)
    lengthS = len(sents)

    long_words = [w for w in word_tokens if len(w) > 7]
    pl = len(long_words) / lengthW * 100  # procent long

    GFI = 0.4 * ((lengthW / lengthS) + pl)
    return  GFI


def fre(text):
    # Flesch Reading Ease
    word_tokens = nltk.tokenize.regexp_tokenize(text, r'\w+')
    lengthW = len(word_tokens)

    sents = sent_tokenizer.tokenize(text)
    lengthS = len(sents)
    ts = 0  # total syllables
    for word in word_tokens:
        ts = ts + sylco(word)

    FRE = 206.835 - 1.015 * (lengthW / lengthS) - 84.6 * (ts / lengthW)
    return FRE


def fkgl(text):
    # Flesch–Kincaid Grade Level
    word_tokens = nltk.tokenize.regexp_tokenize(text, r'\w+')
    lengthW = len(word_tokens)

    sents = sent_tokenizer.tokenize(text)
    lengthS = len(sents)

    ts = 0  # total syllables
    for word in word_tokens:
        ts = ts + sylco(word)

    FKGL = 0.39 * (lengthW / lengthS) + 11.8 * (ts / lengthW) - 15.59
    return FKGL


def dcrf(text):

    word_tokens = nltk.tokenize.regexp_tokenize(text, r'\w+')
    lengthW = len(word_tokens)

    sents = sent_tokenizer.tokenize(text)
    lengthS = len(sents)

    long_words = [w for w in word_tokens if len(w) > 7]
    pl = len(long_words) / lengthW * 100  # procent long

    # Dale–Chall readability formula
    DCRF = 0.1579 * pl + 0.0496 * (lengthW / lengthS)

    return DCRF


def ari(text):
    word_tokens = nltk.tokenize.regexp_tokenize(text, r'\w+')
    lengthW = len(word_tokens)

    sents = sent_tokenizer.tokenize(text)
    lengthS = len(sents)
    lengthCH = len(text)
    ARI = 4.71 * (lengthCH / lengthW) + 0.5 * (lengthW / lengthS) - 21.43
    return ARI


def smog(text):
    word_tokens = nltk.tokenize.regexp_tokenize(text, r'\w+')

    sents = sent_tokenizer.tokenize(text)
    lengthS = len(sents)
    tps = 0  # total syllables
    for word in word_tokens:
        if sylco(word) > 2:
            tps = tps + 1
    SMOG = 1.043 * math.sqrt(tps * (30 / lengthS)) + 3.1291
    return SMOG



def sent_len(text):
    text = nltk.sent_tokenize(text)
    text_lens = [len(nltk.word_tokenize(sent)) for sent in text]
    # print(sum(text_lens)/len(text_lens))
    return sum(text_lens) / len(text_lens)

def ttr(text):
    words = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    return (len(set(words) ) /len(words))




class ComperhensionFeatures:
    def __init__(self,
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
        self.verbose = verbose
        self.methods = {"sylco":sylco, "gfi": gfi, 
                        "fre": fre, "fkgl": fkgl, 
                        "dcrf": dcrf, "ari": ari, 
                        "smog": smog, "sent_len": sent_len, 
                        "ttr": ttr}
        self.ndim = len(self.methods)

    def fit(self, text_list):
        """
        The fit method.

        :param text_list: List of input texts
        
        """
        pass

    def transform(self, new_documents):
        """
        Transform method.

        :param new_documents: The new set of documents to be transformed.
        :return all_embeddings: The final embedding matrix
        
        """

        if not type(new_documents) == list:
            new_documents.values.tolist()

        if self.verbose:
            logging.info("[Comperhension Features] Transforming new documents.")

        new_features = np.zeros((len(new_documents), self.ndim))
        for enx, doc in tqdm.tqdm(enumerate(new_documents),
                                  total=len(new_documents)):
            for mid, method in enumerate(self.methods):
                value = self.methods[method](doc)
                new_features[mid] = value

        return new_features

    def fit_transform(self, documents, b=None):
        """
        The sklearn-like fit-transform method.

        """

        self.fit(documents)
        return self.transform(documents)

    def get_feature_names_out(self):
        """
        Get feature names.
        """

        return list(self.methods.keys())


if __name__ == "__main__":

    example_text = pd.read_csv("../../data/insults/train.tsv", sep="\t")['text_a']
    labels = pd.read_csv("../../data/insults/train.tsv",
                         sep="\t")['label'].values.tolist()
    clx = ComperhensionFeatures()
    sim_features = clx.fit_transform(example_text)

    print(clx.get_feature_names_out())
