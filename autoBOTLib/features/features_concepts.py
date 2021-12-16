
## relation extractor
## https://conceptnet.io/
## https://github.com/commonsense/conceptnet5/wiki/Downloads

import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

import gzip
import os
import pandas as pd
import networkx as nx
import tqdm
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

POSSIBLE_RELATIONS = ["is_a","part_of","subset_of","represents_a","serves_for","lives_in","works_for","represents_the","looks_like","synonym_for","similar_to","context_of","related_to","knows_of","knowledge_of","serves_to","serves_for","lives_near","parent_of","child_of","superior_to"]

class ConceptFeatures:
    """
    Core class describing sentence embedding methodology employed here.
    """
    def __init__(self,
                 max_features=10000,
                 targets=None,
                 knowledge_graph="../memory"):

        self.max_features = max_features
        self.targets = targets
        ## TODO -> download conceptnet if not present.

        self.knowledge_graph = knowledge_graph
        self.feature_names = None

    def get_grounded_from_path(self, present_tokens, graph_path):
        """
        Method which performs a very simple term grounding. This simply evaluates if both terms are present in the corpus.
        :param list present_tokens: The present tokens
        :param str graph_path: Path to the triplet base (compressed)
        """

        with gzip.open(graph_path, "rt", encoding="utf-8") as gp:
            for line in gp:
                subject, predicate, obj = line.strip().split("\t")
                if subject.lower() in present_tokens and obj.lower(
                ) in present_tokens:
                    if subject != obj:
                        yield (subject, predicate, obj)


    def add_triplet(self, tokens, index, relations=["is_a"]):

        token = tokens[index]        
        for relation in relations:
            p1, p2 = relation.split("_")
            if token.lower() == p1 and tokens[index + 1].lower() == p2:
                if len(tokens[index - 1]) > 1 and len(tokens[index + 2]) > 1:
                    triplet_adhoc = (tokens[index - 1], relation,
                                     tokens[index + 2])
                    if triplet_adhoc[0] != triplet_adhoc[2]:
                        if len(triplet_adhoc[0]) > 3 and len(triplet_adhoc[2]) > 3:
                            yield triplet_adhoc
        
                        
    def concept_graph(self, document_space, graph_path):
        """
        If no prior knowledge graph is supplied, one is constructed.
        :param document_space: The list of input documents
        :param graph_path: The path of the knowledge graph used.
        :return grounded: Grounded relations.
        """

        generic_triplets = []
        present_tokens = set()

        for document in document_space:
            tokens = nltk.word_tokenize(document)
            tokens = [word.lower() for word in tokens]
            for i, token in enumerate(tokens):
                if i > 0 and i < len(tokens) - 1:
                    found_triplets = self.add_triplet(tokens, i, relations=POSSIBLE_RELATIONS)
                    for triplet in found_triplets:
                        generic_triplets.append(triplet)

            for token in tokens:
                present_tokens.add(token)

        grounded = []

        if len(generic_triplets) == 0:
            logging.info("Generating the knowledge graph (ad hoc)")
            for document in document_space:
                tokens = nltk.word_tokenize(document)
                tokens = [str(x) for x in tokens]
                for enx, token in enumerate(tokens):
                    if enx > 1 and enx < len(tokens) - 2:
                        if len(tokens[enx - 1]) >= 2 and len(
                                tokens[enx + 1]) >= 2:
                            triplet_adhoc = (tokens[enx - 2], token,
                                             tokens[enx + 2])
                            if triplet_adhoc[0] != triplet_adhoc[2]:
                                generic_triplets.append(triplet_adhoc)

        logging.info(f"Found the following relations: {set([x[1] for x in generic_triplets])}")
        try:

            kg_sources = os.listdir(graph_path)
            full_paths = [os.path.join(graph_path, x) for x in kg_sources]

            for path in full_paths:
                triplet_generator = self.get_grounded_from_path(
                    present_tokens, path)

                for triplet in triplet_generator:
                    grounded.append(triplet)
                logging.info(f"Grounded in total: {len(grounded)} triplets (last added from {path}).")

        except Exception as es:
            logging.info(
                f"No knowledge graphs found in the default path: {graph_path}. Reverting to generic triplet extraction from the corpus alone. To use a knowledge graph, place a Gzipped triplet (tsv) database in {graph_path} folder. ({es})"
            )
            grounded = generic_triplets
            del generic_triplets

        logging.info(
            "The number of grounded relations in the input corpus is: {}".
            format(len(grounded)))

        if len(grounded) == 0:
            logging.info(
                "No grounded relations found, please recoonsider the used knowledge base."
            )

        return grounded

    def get_propositionalized_rep(self, documents):
        """
        The method for constructing the representation.

        :param documents: The input list of documents.
        """

        G = nx.DiGraph()
        for (s, p, o) in tqdm.tqdm(self.grounded_triplets):
            G.add_edge(s, o, type=p)
        numbered_links = {e: enx for enx, e in enumerate(G.edges())}

        if self.feature_names is None:
            self.feature_names = [
                e[2]['type'] + "(" + e[0] + "," + e[1] + ")"
                for e in G.edges(data=True)
            ]
            self.feature_types = [e[2]['type'] for e in G.edges(data=True)]
        rbags = []
        all_doc_sizes = []
        self.t_counter = Counter()
        for enx2, document in tqdm.tqdm(enumerate(documents),
                                        total=len(documents)):
            tokens = nltk.word_tokenize(document)
            tokens = [word.lower() for word in tokens]
            doc_tokens = set(tokens)
            present = doc_tokens.intersection(G.nodes())
            subgraph = G.subgraph(present)
            rb = []

            for edge in subgraph.edges(data=True):

                enx1 = numbered_links[(edge[0], edge[1])]
                fname = edge[2]['type'] + "(" + edge[0] + "," + edge[1] + ")"
                self.t_counter.update({fname: 1})
                enx1 = self.feature_names[enx1]
                rb.append(enx1)

            all_doc_sizes.append(len(rb))
            rbags.append(" ".join(rb))
        logging.info("Most common relations below.")
        logging.info(self.t_counter.most_common(10))
        logging.info(f"Average bag size: {np.mean(all_doc_sizes)}")
        return rbags

    def fit(self, text_vector, refit=False, knowledge_graph=None):
        """
        Fit the model to a text vector.
        
        :param text_vector: Input list of documents.
        """

        if self.knowledge_graph is None:
            self.knowledge_graph = knowledge_graph

        logging.info("Constructing the token graph.")
        sentences_separated = []
        for doc in text_vector:
            sentences = nltk.tokenize.sent_tokenize(doc)
            for els in sentences:
                sentences_separated.append(els)
        self.grounded_triplets = self.concept_graph(sentences_separated,
                                                    self.knowledge_graph)
        logging.info("Relation propositionalization ..")

        # this identifies relational bags that are grounded + stores the info
        self.conc_docs = self.get_propositionalized_rep(text_vector)
        logging.info("Concept-based features extracted.")

        self.concept_vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),
            sublinear_tf=True,  ## for very sparse spaces
            max_features=self.max_features,
            token_pattern=r'\S+').fit(self.conc_docs)

    def transform(self, text_vector, use_conc_docs=False):
        """
        Transform the data into suitable form.
        """

        if use_conc_docs:
            text_vector = self.conc_docs
        else:
            text_vector = self.get_propositionalized_rep(text_vector)
        return self.concept_vectorizer.transform(text_vector)

    def get_feature_names(self):

        return self.concept_vectorizer.get_feature_names()

    def fit_transform(self, text_vector, b=None):
        """
        A classifc fit-transform method.

        :param text_vector: The input list of documents.
        :return transformedObj: Transformed texts (to features).
        """

        try:
            self.fit(text_vector, self.targets)
        except Exception as es:
            logging.info(es)
            
        return self.transform(text_vector,
                              use_conc_docs=True)  # use stored docs


if __name__ == "__main__":

    example_text = pd.read_csv("../data/depression/train.tsv", sep="\t")
    text = example_text['text_a']
    labels = example_text['label']
    rex = ConceptFeatures(knowledge_graph="./memory")
    m = rex.fit_transform(text)
    fnames = rex.get_feature_names()

    m = m.todense()
    dataframe = pd.DataFrame(m)
    dataframe.columns = fnames
    dataframe['target'] = labels
    dataframe.to_csv("patronizing.tsv", sep = "\t", index=False)
