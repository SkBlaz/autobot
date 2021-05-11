
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

import pandas as pd
import scipy.io
import numpy as np
from scipy.sparse import csgraph,csr_matrix
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import tqdm
from collections import OrderedDict
import networkx as nx

class RelationalDocs:

    def __init__(self, ndim = 128,
                 random_seed = 1965123,
                 ed_cutoff = -2, verbose = True,
                 percentile_threshold = 95):

        """
        Class initialization method.

        :param ndim: Number of latent dimensions
        :param random_seed: The random seed used
        :param ed_cutoff: Cutoff for fuzzy string matching when comparing documents
        :param verbose: Whether to have the printouts
        
        """
        
        self.ndim = ndim
        self.percentile_threshold = percentile_threshold
        self.verbose = verbose
        self.random_seed = random_seed
        self.ed_cutoff = ed_cutoff

    def jaccard_index(self, set1, set2):

        """
        The classic Jaccard index.
        
        :param set1: First set
        :param set2: Second set
        :return JaccardIndex:
        """

        intersection = set1.intersection(set2)
        common_space = set.union(set1, set2)
        return len(intersection)/(len(common_space)+1)
        
    def fit(self, text_list):

        """
        The fit method.

        :param text_list: List of input texts
        
        """

        if not type(text_list) == list:
            text_list = text_list.values.tolist()

        all_weights = []
        t_tokens = {a:set([x.lower()[:self.ed_cutoff] for x in a.strip().split(" ")]) for a in text_list}
        t_tokens = OrderedDict(t_tokens)
        nlist = {}
        
        for a in tqdm.tqdm(range(len(text_list))):
            for b in range(a, len(text_list)):
                set1 = t_tokens[text_list[a]]
                set2 = t_tokens[text_list[b]]
                jaccard = self.jaccard_index(set1, set2)
                nlist[(a,b)] = jaccard
                
        self.core_documents = t_tokens
        self.G = self.get_graph(nlist, len(text_list))
        G = nx.to_scipy_sparse_matrix(self.G, nodelist = list(range(len(text_list))))
        
        if self.verbose:
            logging.info("Graph normalization in progress.")
            
        laplacian = csgraph.laplacian(G, normed=True)

        if self.verbose:
            logging.info("SVD of the graph relation space in progress.")

        if self.ndim >= len(text_list): self.ndim = len(text_list)-1
            
        svd = TruncatedSVD(n_components=self.ndim, random_state=self.random_seed)
        
        self.node_embeddings = svd.fit_transform(laplacian)
        self.neigh_size = int(np.cbrt(self.node_embeddings.shape[0]))

    def transform(self, new_documents):

        """
        Transform method.

        :param new_documents: The new set of documents to be transformed.
        :return all_embeddings: The final embedding matrix
        
        """

        if not type(new_documents) == list:
            text_list = new_documents.values.tolist()

        if self.verbose:
            logging.info("Transforming new documents.")
            
        all_embeddings = []
        for doc in tqdm.tqdm(new_documents):
            doc_split = set([x.lower()[:self.ed_cutoff] for x in doc.strip().split(" ")])
            distances = []
            for k, v in self.core_documents.items():
                dist = self.jaccard_index(doc_split, v)
                distances.append(dist)
            distances = np.array(distances)
            sorted_dists = np.argsort(distances)[::-1]
            local_neigh_size = self.neigh_size
            n_iter = 1000
            ni = 0
            embedding = np.mean(self.node_embeddings[sorted_dists[0:local_neigh_size]], axis = 0)                
            all_embeddings.append(embedding)
        all_embeddings = np.array(all_embeddings)
        assert len(new_documents) == all_embeddings.shape[0]
        return all_embeddings

    def fit_transform(self, documents, b = None):

        """
        The sklearn-like fit-transform method.

        """
        
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self):        
        return list(["dim_"+str(x) for x in range(self.ndim)])

    def get_graph(self, wspace, ltl):

        """
        A method to obtain a graph from a weighted space of documents.
        
        :param wspace: node1,node2 weight mapping
        :param ltl: The number of documents
        :return G: The document graph

        """
        
        valspace = np.sort([w for w in wspace.values() if w > 0])
        max_val = max(valspace)
        lnode = None
        
        if self.verbose:
            logging.info("Obtaining the document graph.")
            
        perc = np.percentile(valspace, self.percentile_threshold)
        subset_edges = [k for k, v in wspace.items() if v > perc]
        G = nx.Graph()

        for el in subset_edges:
            G.add_edge(el[0],el[1], weight = wspace[el[0], el[1]])

        if self.verbose:
            logging.info("Obtained the document graph")
        
        return G

if __name__ == "__main__":
    
    example_text = pd.read_csv("../data/hatespeech/train.tsv", sep="\t")['text_a']
    labels = pd.read_csv("../data/hatespeech/train.tsv", sep="\t")['label'].values.tolist()
    clx = RelationalDocs(percentile_threshold = 90, ed_cutoff = -2)
    sim_features = clx.fit_transform(example_text)

    import matplotlib.pyplot as plt
    import seaborn as sns
    import operator
    
    from sklearn.linear_model import LogisticRegression
    from sentence_embeddings import *
    from sklearn.model_selection import cross_val_score
    from sklearn.dummy import DummyClassifier

    # dem = documentEmbedder()
    # dem_features = dem.fit_transform(example_text).todense()

    # clf = LogisticRegression()
    # lc = labels.copy()
    # cross_val_score = cross_val_score(clf, dem_features, lc, cv = 5)
    # print(cross_val_score, "doc2vec")

    # clf = LogisticRegression()
    # lc = labels.copy()
    # cross_val_score = cross_val_score(clf, sim_features, lc, cv = 5)
    # print(cross_val_score, "dsim")

    clf = DummyClassifier()
    cross_val_score = cross_val_score(clf, sim_features, labels.copy(), cv = 5)
    print(cross_val_score, "dummy")
    
    print("Plotting")
    doc_graph = clx.G
    prx = nx.pagerank(doc_graph)
    ranks = [y for _,y in prx.items()]
    colors = []
    mdoc = list(sorted(prx.items(), key=operator.itemgetter(1)))[-10:]
    for ex in mdoc:
        print(example_text[ex[0]], ex[1])

    print(nx.info(doc_graph))
    node_sizes = []
    for n in doc_graph.nodes():
        colors.append(prx[n])
        if prx[n] != max(ranks):
            node_sizes.append(6)
        else:
            node_sizes.append(10)

    pos = nx.spring_layout(doc_graph,scale=2, iterations = 100)
    nx.draw_networkx_nodes(doc_graph, pos, node_size = node_sizes, node_color = colors)
    nx.draw_networkx_edges(doc_graph, pos, alpha = 0.1)
    plt.show()
