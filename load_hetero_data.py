import os
import io
import sys
import numpy as np
import pickle as pkl
from pgl.graph import Graph
import networkx as nx
import scipy.sparse as sp


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = (rowsum == 0) * 1 + rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


class HeteroDataset(object):

    def __init__(self, name, splits_file_path=None, symmetry_edges=True, self_loop=False):
        self.splits_file_path = splits_file_path
        self.symmetry_edges = symmetry_edges  # covert to undirectional graph
        self.self_loop = self_loop  # add self-loop
        self.name = name
        self._load_data()

    def _load_data(self):
        """Load data
        """
        graph_adjacency_list_file_path = os.path.join('dataset', self.name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('dataset', self.name, 'out1_node_feature_label.txt')

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}
        if self.name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])

        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        # print(graph_node_features_dict)

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        # adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
        features = preprocess_features(features)
        features = np.array(features, dtype="float32")

        all_edges = []
        for i in G.edges():
            u, v = tuple(i)
            all_edges.append((u, v))
            if self.symmetry_edges:
                all_edges.append((v, u))

        if self.self_loop:
            for i in range(_graph.number_of_nodes()):
                all_edges.append((i, i))

        all_edges = list(set(all_edges))

        with np.load(self.splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

        self.graph = Graph(num_nodes=G.number_of_nodes(), edges=all_edges, node_feat={"feat": features})
        self.y = np.array(labels, dtype="int64")
        self.num_classes = int(max(labels)) + 1
        idx_train = []
        idx_val = []
        idx_test = []

        for i in range(len(train_mask)):
            if train_mask[i]:
                idx_train.append(i)
            elif val_mask[i]:
                idx_val.append(i)
            elif test_mask[i]:
                idx_test.append(i)

        self.train_index = np.array(idx_train, dtype="int32")
        self.val_index = np.array(idx_val, dtype="int32")
        self.test_index = np.array(idx_test, dtype="int32")
