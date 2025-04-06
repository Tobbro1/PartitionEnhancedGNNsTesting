# Implements extraction of dataset properties and index lookup functionality. This can be used in both the generation and interpretation of vertex partition features.

# general imports
from typing import Optional, Tuple, List
import json
import pickle
import os.path as osp
import os
import time

# numpy
import numpy as np

# pytorch
import torch
from torch import Tensor

# pytorch geometric
from torch_geometric.data import Dataset
from torch_geometric.utils import to_networkx

# networkx
import networkx as nx

class Dataset_Properties_Manager():
    # Manages dictionaries for 
    #   - properties of the dataset and their computation
    #   - index lookup on the given dataset with a given sample indexing

    # Requires either a path from which to read a properties json file or a dataset
    # NOTE: If this Manager is utilised for index conversion of an already computed feature vector dataset, it is 
    def __init__(self, absolute_path_prefix: str, properties_path: Optional[str] = None, dataset: Optional[Dataset] = None, node_pred: bool = False, write_properties_root_path: Optional[str] = None, write_properties_filename: Optional[str] = None):
        super().__init__()

        self.node_pred = node_pred

        assert properties_path is not None or dataset is not None

        self.absolute_path_prefix = absolute_path_prefix
        self.properties_file_path = properties_path

        # Properties about the dataset that should be stored
        # Task - node prediction or graph prediction
        # num_graphs total
        # num_vertices total
        # num_labels total
        # num_classes
        # graph_size - avg, min, max
        # Diameter of graphs - avg, min, max
        # num_labels per graph - avg, min, max
        # computation time of the properties
        # label_alphabet
        # graph_sizes
        self.properties = {}

        if properties_path is not None:
            path = osp.join(self.absolute_path_prefix, properties_path)

            # Read the properties from disk
            if not osp.exists(path):
                raise FileNotFoundError
            else:
                with open(path, "r") as file:
                    self.properties = json.loads(file.read())
                
        else:
            # Generate properties from the dataset
            t0 = time.time()

            # Initialize for proper order
            if node_pred:
                self.properties["task"] = "node classification"
            else:
                self.properties["task"] = "graph classification"
            self.properties["num_graphs"] = dataset.len()
            self.properties["num_vertices"] = 0
            self.properties["num_labels"] = 0
            self.properties["num_classes"] = dataset.num_classes
            self.properties["graph_size"] = {}
            self.properties["diameter"] = {}
            self.properties["labels_per_graph"] = {}
            self.properties["time"] = float(-1)
            self.properties["node_pred"] = node_pred 
            self.properties["label_alphabet"] = []
            self.properties["graph_sizes"] = []

            if dataset.num_features > 1:
                # assumed to be one-hot-encoding
                self.properties["label_alphabet"] = list(range(dataset.num_features))

            min_size = float('inf')
            max_size = -1
            min_diameter = float('inf')
            max_diameter = -1
            sum_diameter = 0
            min_labels = float('inf')
            max_labels = -1
            sum_labels = 0

            for graph_id in range(self.properties["num_graphs"]):
                cur_graph = dataset.get(graph_id)

                # Graph size
                cur_graph_num_vertices = cur_graph.x.shape[0]

                if cur_graph_num_vertices < min_size:
                    min_size = cur_graph_num_vertices
                if cur_graph_num_vertices > max_size:
                    max_size = cur_graph_num_vertices

                self.properties["num_vertices"] += cur_graph_num_vertices
                self.properties["graph_sizes"].append(cur_graph_num_vertices)

                # Diameter
                diameter = nx.diameter(G = to_networkx(data = cur_graph))
                if diameter < min_diameter:
                    min_diameter = diameter
                if diameter > max_diameter:
                    max_diameter = diameter
                sum_diameter += diameter

                # Labels
                if dataset.num_features == 1:
                    # Assumed to be labeling
                    unique_labels_tensor = torch.unique(input = cur_graph.x, sorted = False)
                    _new_labels = unique_labels_tensor.flatten().tolist()
                    num_new_labels = len(_new_labels)

                    if num_new_labels < min_labels:
                        min_labels = num_new_labels
                    if num_new_labels > max_labels:
                        max_labels = num_new_labels
                    sum_labels += num_new_labels

                    self.properties["label_alphabet"].extend(_new_labels)
                    # Remove duplicates added
                    self.properties["label_alphabet"] = list(set(self.properties["label_alphabet"]))
                elif dataset.num_features > 1:
                    # Untested

                    # Assumed to be one-hot encoding
                    unique_labels_tensor = torch.unique(input = cur_graph.x, sorted = False, dim = 1)
                    num_new_labels = unique_labels_tensor.size(dim = 0)

                    if num_new_labels < min_labels:
                        min_labels = num_new_labels
                    if num_new_labels > max_labels:
                        max_labels = num_new_labels
                    sum_labels += num_new_labels

                    # Only set this once
                    if graph_id == 0:
                        self.properties["label_alphabet"] = list(range(dataset.num_features))

            self.properties["num_labels"] = len(set(self.properties["label_alphabet"]))

            # Compute averages
            avg_size = float(self.properties["num_vertices"]) / self.properties["num_graphs"]
            avg_diameter = float(sum_diameter) / self.properties["num_graphs"]
            avg_labels = float(sum_labels) / self.properties["num_graphs"]

            # Store the avg, min, max properties
            self.properties["graph_size"]["avg"] = avg_size
            self.properties["graph_size"]["min"] = min_size
            self.properties["graph_size"]["max"] = max_size

            self.properties["diameter"]["avg"] = avg_diameter
            self.properties["diameter"]["min"] = min_diameter
            self.properties["diameter"]["max"] = max_diameter

            self.properties["labels_per_graph"]["avg"] = avg_labels
            self.properties["labels_per_graph"]["min"] = min_labels
            self.properties["labels_per_graph"]["max"] = max_labels

            # Store the computation time of the features
            self.properties["time"] = time.time() - t0

            if write_properties_root_path is not None:
                # Write the computed properties into a file
                assert write_properties_filename is not None

                path = osp.join(self.absolute_path_prefix, write_properties_root_path)

                if not osp.exists(path):
                    os.makedirs(path)

                p = osp.join(path, write_properties_filename)
                if not osp.exists(p):
                    open(p, 'w').close()  

                with open(p, "w") as file:
                    file.write(json.dumps(self.properties, indent=4))

                self.properties_file_path = osp.join(write_properties_root_path, write_properties_filename)
            else:
                self.properties_file_path = None

        # convert self.properties["graph_sizes"] to a numpy array for easier indexing later
        self.properties["graph_sizes"] = np.array(self.properties["graph_sizes"])


    # Samples specifies the vertex_ids or graph_ids (depending on node_pred), for which the index lookup functions are computed
    # Returns a list of vertex indices that are considered in samples and graph indexes that are considered in graph_samples. Additionally returns the path of the pkl file where the lookups are stored on disk.
    def initialize_idx_lookups(self, lookup_path: Optional[str], samples: Optional[List[int]] | Optional[Tensor] = None, write_lookup_root_path: Optional[str] = None, write_lookup_filename: Optional[str] = None) -> Tuple[List[int], List[int], str]:

        lookup_disk_path = None

        if lookup_path is not None:
            path = osp.join(self.absolute_path_prefix, lookup_path)
            if not osp.exists(path):
                raise FileNotFoundError
            else:
                with open(path, 'rb') as file:
                    self.dataset_idx_lookup, self.database_graph_start_idx, samples, graph_samples = pickle.load(file = file)
                    self.samples = list(samples)
                    self.graph_samples = list(graph_samples)
                    lookup_disk_path = lookup_path
        else:
            # generate lookup dictionaries and samples, graph_samples lists

            # A dictionary utilised to speed up index computation methods. 
            # In the node prediction setting this corresponds to a map dataset_idx -> database_idx with entries for every dataset_idx contained in samples.
            # In the graph prediction setting this corresponds to a map dataset_idx -> graph_id, vertex_id with entries for every dataset_idx contained in the graphs in samples
            self.dataset_idx_lookup = {}

            # A dictionary containing key-value pairs of the form {graph_id: database_start_idx} giving the starting idx of vertices of the given graph in the database. 
            # This does not correspond to the idx in the dataset. This is used to efficiently compute a database_vertex_idx from a graph_id.
            # Only utilised in the graph prediction setting
            self.database_graph_start_idx = {}

            if samples is None:
                # We compute the feature vectors for the whole dataset
                self.samples = []
                self.graph_samples = []

                if self.node_pred:
                    # One graph, untested
                    for dataset_idx in range(self.properties["num_vertices"]):
                        self.dataset_idx_lookup[dataset_idx] = dataset_idx
                        self.samples.append(dataset_idx)
                    self.graph_samples = [0]
                else:
                    # Multiple graphs
                    self.database_graph_start_idx = {}

                    self.graph_samples.append(0)
                    graph_id = 0
                    cur_start_idx = 0
                    self.database_graph_start_idx[0] = 0
                    for dataset_idx in range(self.properties["num_vertices"]):
                        if (dataset_idx - cur_start_idx) == int(self.properties["graph_sizes"][graph_id]):
                            graph_id += 1
                            self.graph_samples.append(graph_id)
                            cur_start_idx = dataset_idx
                            self.database_graph_start_idx[graph_id] = dataset_idx

                        self.dataset_idx_lookup[dataset_idx] = tuple([graph_id, dataset_idx - cur_start_idx])
                        self.samples.append(dataset_idx)
            else:
                # samples is given, in the node prediction setting it is a list of vertex_ids, in the graph prediction setting it is a list of graph_ids

                if torch.is_tensor(samples):
                    samples = samples.tolist()

                assert len(samples) > 0

                if self.node_pred:
                    # One graph
                    self.graph_samples = [0]
                    self.samples = []

                    # Pre-compute the dataset_idx_lookup dictionary, a map representing dataset_idx -> database_idx
                    for dataset_idx in samples:
                        self.dataset_idx_lookup[dataset_idx] = len(samples)
                        self.samples.append(dataset_idx)
                else:
                    # Multiple graphs
                    self.graph_samples = samples
                    self.samples = []

                    self.database_graph_start_idx = {}

                    # We compute a list of the vertices included in the graphs (defined by the graph_ids in samples). Additionaly we store the start_idx property to efficiently compute the inverse of this operation via look-up.
                    for graph_id in self.graph_samples:
                        start_idx = int(sum(self.properties["graph_sizes"][:graph_id]))

                        # Store database_start_idx for a given graph. This is given by the length of the sample list before adding the vertices of the given graph.
                        assert graph_id not in self.database_graph_start_idx
                        self.database_graph_start_idx[graph_id] = len(self.samples)

                        # Refine samples
                        num_nodes = int(self.properties["graph_sizes"][graph_id])
                        self.samples.extend(list(range(start_idx, start_idx + num_nodes)))

                        # Pre-compute the dataset_idx_lookup dictionary
                        for dataset_idx in range(start_idx, start_idx + num_nodes):
                            self.dataset_idx_lookup[dataset_idx] = tuple([graph_id, dataset_idx - start_idx])

            if write_lookup_root_path is not None:
                # Write the computed lookups as well as samples and graph_samples into a file
                assert write_lookup_filename is not None

                path = osp.join(self.absolute_path_prefix, write_lookup_root_path)
                if not osp.exists(path):
                    os.makedirs(path)

                p = osp.join(path, write_lookup_filename)
                if not osp.exists(p):
                    open(p, 'wb').close()  

                with open(p, 'wb') as file:
                    samples_tuple = tuple(self.samples)
                    graph_samples_tuple = tuple(self.graph_samples)
                    pickle_tuple = tuple([self.dataset_idx_lookup, self.database_graph_start_idx, samples_tuple, graph_samples_tuple])
                    pickle.dump(obj = pickle_tuple, file = file)
                
                lookup_disk_path = osp.join(write_lookup_root_path, write_lookup_filename)

        assert self.dataset_idx_lookup is not None and self.database_graph_start_idx is not None and self.samples is not None and self.graph_samples is not None

        return self.samples, self.graph_samples, lookup_disk_path
    
    # Helper to get the graph_id and vertex_id from a dataset idx. NOTE: The dataset idx is the idx used in the dataset, not in the tensor used to store results (which would be the database idx).
    def get_vertex_identifier_from_dataset_idx(self, dataset_idx: int) -> Tuple[int, int]:
        
        if self.node_pred:
            return tuple([0, dataset_idx])
        else:
            # We use a pre-computed dictionary representing dataset_idx -> graph_id, vertex_id to speed up the computation
            return self.dataset_idx_lookup[dataset_idx]
        
    # NOTE: In the node prediction scenario this method currently does not support computing a sample of the vertices of the dataset.
    def get_database_idx_from_vertex_identifier(self, vertex_identifier: Tuple[int,int]) -> int:   
    
        if self.node_pred:
            # Only one graph
            # vertex_identifier[1] is the dataset_idx of the given vertex. We use dataset_idx_lookup; a dictionary representing a map dataset_idx -> database_idx directly
            return self.dataset_idx_lookup[vertex_identifier[1]]
        else:
            # Multiple graphs, vertex_identifier[0] represents the graph_id and vertex_identifier[1] represents the idx of the vertex within the graph. Utilise graph_start_idx as a lookup method
            return self.database_graph_start_idx[vertex_identifier[0]] + vertex_identifier[1]
        
    def get_dataset_idx_from_vertex_identifier(self, vertex_identifier: Tuple[int,int]) -> int:
        return int(sum(self.properties["graph_sizes"][:vertex_identifier[0]])) + vertex_identifier[1]