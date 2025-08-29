import math
import re
import pandas as pd
import numpy as np
import tqdm
import nltk 
from nltk import sent_tokenize, regexp_tokenize

import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

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
    double_and_triple = len(re.findall(r'[eaoui][eaoui]', word))
    tripple = len(re.findall(r'[eaoui][eaoui][eaoui]', word))
    disc += double_and_triple + tripple

    # 5) count remaining vowels in word.
    num_vowels = len(re.findall(r'[eaoui]', word))

    # 9) if starts with "tri-" or "bi-" and is followed by a vowel, add one.
    if word[:3] == "tri" and len(word) > 3 and word[3] in "aeoui":
        syls += 1

    if word[:2] == "bi" and len(word) > 2 and word[2] in "aeoui":
        syls += 1

    # 10) if ends with "-ian", should be counted as two syllables, except for "-tian" and "-cian"
    # 13) check for "-n't" and cross match with dictionary to add syllable.
    # (These rules would be added if needed.)

    # calculate the output
    return num_vowels - disc + syls


def gfi(text):
    # Gunning Fog Index
    word_tokens = regexp_tokenize(text, r'\w+')
    length_w = len(word_tokens)
    sents = sent_tokenize(text)
    length_s = len(sents)

    if length_s == 0 or length_w == 0:
        return 0

    long_words = [w for w in word_tokens if len(w) > 7]
    pl = len(long_words) / length_w * 100  # percentage long words
    gfi = 0.4 * ((length_w / length_s) + pl)
    return gfi


def fre(text):
    # Flesch Reading Ease
    word_tokens = regexp_tokenize(text, r'\w+')
    length_w = len(word_tokens)
    sents = sent_tokenize(text)
    length_s = len(sents)

    if length_s == 0 or length_w == 0:
        return 0

    ts = 0  # total syllables
    for word in word_tokens:
        ts += sylco(word)

    fre = 206.835 - 1.015 * (length_w / length_s) - 84.6 * (ts / length_w)
    return fre


def fkgl(text):
    # Flesch–Kincaid Grade Level
    word_tokens = regexp_tokenize(text, r'\w+')
    length_w = len(word_tokens)
    sents = sent_tokenize(text)
    length_s = len(sents)

    if length_s == 0 or length_w == 0:
        return 0

    ts = 0  # total syllables
    for word in word_tokens:
        ts += sylco(word)

    fkgl = 0.39 * (length_w / length_s) + 11.8 * (ts / length_w) - 15.59
    return fkgl


def dcrf(text):
    # Dale–Chall Readability Formula
    word_tokens = regexp_tokenize(text, r'\w+')
    length_w = len(word_tokens)
    sents = sent_tokenize(text)
    length_s = len(sents)

    if length_s == 0 or length_w == 0:
        return 0

    long_words = [w for w in word_tokens if len(w) > 7]
    pl = len(long_words) / length_w * 100  # percentage of long words

    dcrf = 0.1579 * pl + 0.0496 * (length_w / length_s)
    return dcrf


def ari(text):
    # Automated Readability Index
    word_tokens = regexp_tokenize(text, r'\w+')
    length_w = len(word_tokens)
    sents = sent_tokenize(text)
    length_s = len(sents)
    length_ch = len(text)

    # Avoid division by zero
    if length_w == 0 or length_s == 0:
        return 0

    ari = 4.71 * (length_ch / length_w) + 0.5 * (length_w / length_s) - 21.43
    return ari


def smog(text):
    # SMOG Index
    word_tokens = regexp_tokenize(text, r'\w+')
    sents = sent_tokenize(text)
    length_s = len(sents)

    if length_s == 0:
        return 0

    tps = 0  # total words with more than 2 syllables
    for word in word_tokens:
        if sylco(word) > 2:
            tps += 1

    SMOG = 1.043 * math.sqrt(tps * (30 / length_s)) + 3.1291
    return SMOG


def sent_len(text):
    sents = nltk.sent_tokenize(text)
    if not sents:
        return 0
    # Calculate average sentence length (in words)
    text_lens = [len(nltk.word_tokenize(sent)) for sent in sents]
    if len(text_lens) == 0:
        return 0
    return sum(text_lens) / len(text_lens)


def ttr(text):
    # Type-Token Ratio (vocabulary diversity)
    words = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)


class ComperhensionFeatures:
    def __init__(self,
                 verbose=True):
        """
        Class initialization method.

        :param verbose: Whether to have the printouts
        
        """
        self.verbose = verbose
        self.features = {"gfi": gfi, 
                        "fre": fre, "fkgl": fkgl, 
                        "dcrf": dcrf, "ari": ari, 
                        "smog": smog, "sent_len": sent_len, 
                        "ttr": ttr}
        self.ndim = len(self.features)

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

        if type(new_documents) is not list:
            new_documents.values.tolist()

        if self.verbose:
            logging.info("[Comperhension Features] Transforming new documents.")

        new_features = np.zeros((len(new_documents), self.ndim))
        for enx, doc in tqdm.tqdm(enumerate(new_documents),
                                  total=len(new_documents)):
            for mid, method in enumerate(self.features):
                value = self.features[method](doc)
                if mid < new_features.shape[1]:  # Check column bounds
                    new_features[enx, mid] = value

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

        return list(self.features.keys())


if __name__ == "__main__":

    example_text = pd.read_csv("../../data/insults/train.tsv", sep="\t")['text_a']
    labels = pd.read_csv("../../data/insults/train.tsv",
                         sep="\t")['label'].values.tolist()
    clx = ComperhensionFeatures()
    sim_features = clx.fit_transform(example_text)

    print(clx.get_feature_names_out())
