import logging
import random
import networkx as nx
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)


class WordGraph:
    def __init__(self, fast=True, verbose=True, window_size=2, sample_ratio=0.1, repeats=10):
        """
        Class initialization method.

        :param fast: Perform Monte-Carlo style feature estimation
        :param window_size: what is the size for the word co-occurance graphs
        :param sample_ratio: the percentage of nodes to sample for MCMS estimation
        :param repeats: 
        
        """
        self.verbose = verbose
        self.fast = fast
        self.window_size = window_size
        self.sample_ratio = sample_ratio
        self.repeats = repeats if fast else 1
        self.features = None

    def fit(self, text_list):
        pass

    def transform(self, new_documents):
        if not isinstance(new_documents, list):
            new_documents = new_documents.values.tolist()
        if self.verbose:
            logging.info("[Network Features] Transforming new documents.")
        data = []
        for text in tqdm(new_documents):
            tokens = self.preprocess_text(text)
            G = self.build_cooccurrence_graph(tokens, self.window_size)
            if len(list(G.nodes)) < 2:
                data.append(self.empty_features())
                continue
            repeated_metrics = [
                self.compute_fast_metrics(self.sample_subgraph(G, self.sample_ratio))
                for _ in range(self.repeats)
            ]
            if self.repeats > 1:
                aggregated = {}
                for key in repeated_metrics[0]:
                    values = [metrics[key] for metrics in repeated_metrics]
                    aggregated[f"{key}_avg"] = np.mean(values)
                    aggregated[f"{key}_std"] = np.std(values)
                    aggregated[f"{key}_min"] = np.min(values)
                    aggregated[f"{key}_max"] = np.max(values)
                data.append(list(aggregated.values()))
                self.features = list(aggregated.keys())
            else:
                metrics = repeated_metrics[0]
                data.append(list(metrics.values()))
                self.features = list(metrics.keys())
        
        return np.array(data)

    def fit_transform(self, documents, b=None):
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names_out(self):
        return self.features

    def build_cooccurrence_graph(self, tokens, window_size):
        G = nx.Graph()
        for i, word in enumerate(tokens):
            for j in range(i + 1, min(i + window_size, len(tokens))):
                G.add_edge(word, tokens[j])
        return G

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in tokens if word.isalnum()]

    def sample_subgraph(self, G, sample_ratio):
        nodes = list(G.nodes)
        sample_size = int(sample_ratio * len(nodes))
        sample_size = max(1, min(sample_size, len(nodes)))
        sampled_nodes = random.sample(nodes, sample_size)
        return G.subgraph(sampled_nodes)

    def compute_fast_metrics(self, G):
        num_nodes = len(G.nodes)
        num_edges = len(G.edges)
        density = nx.density(G)
        avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0
        avg_clustering = nx.average_clustering(G) if num_nodes > 0 else 0

        if num_nodes > 0:
            betweenness = nx.betweenness_centrality(G, normalized=True)
            avg_betweenness = np.mean(list(betweenness.values()))
        else:
            avg_betweenness = 0

        if num_nodes > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest_cc)
            closeness = nx.closeness_centrality(subG)
            avg_closeness = np.mean(list(closeness.values()))
        else:
            avg_closeness = 0

        num_components = nx.number_connected_components(G)

        if num_nodes > 0:
            largest_cc_size = max(len(comp) for comp in nx.connected_components(G))
        else:
            largest_cc_size = 0

        if num_nodes > 0:
            pagerank = nx.pagerank(G)
            avg_pagerank = np.mean(list(pagerank.values()))
            max_pagerank = np.max(list(pagerank.values()))
        else:
            avg_pagerank = 0
            max_pagerank = 0

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": density,
            "avg_degree": avg_degree,
            "avg_clustering": avg_clustering,
            "avg_betweenness": avg_betweenness,
            "avg_closeness": avg_closeness,
            "num_components": num_components,
            "largest_cc_size": largest_cc_size,
            "avg_pagerank": avg_pagerank,
            "max_pagerank": max_pagerank,
        }

    def empty_features(self):
        return [0] * 11 * (4 if self.fast else 1)


if __name__ == "__main__":
    df = pd.read_csv("../../data/insults/train.tsv", sep="\t")
    example_text = df["text_a"]
    labels = df["label"].tolist()
    clx = WordGraph(fast=True, window_size=2, sample_ratio=0.1, repeats=10)
    sim_features = clx.fit_transform(example_text)
    print(clx.get_feature_names_out())
    print(sim_features.shape)
