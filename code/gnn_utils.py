# general
from typing import Optional, Dict, Tuple, List
import time
import json
import os.path as osp

# numpy
import numpy as np

# pytorch
import torch
from torch import Tensor

# pytorch geometric
from torch_geometric.data import Dataset, Data

# scipy
from scipy.sparse import spmatrix

# scikit learn
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import load_svmlight_file

# own functionality
from clustering import Clustering_Algorithm
from dataset_property_util import Dataset_Properties_Manager
from k_disk_sp_feature_generator import compute_single_k_disk_sp_feature_vector
from r_s_ring_sp_feature_generator import compute_single_r_s_ring_sp_feature_vector
from vertex_sp_feature_generator import compute_vertex_sp_feature_vectors
from SP_features import SP_graph_features, SP_vertex_features
import util

# Implements functionality utilised in partition enhanced gnns

# Transforms the features of the given dataset to include the cluster_id so they can be utilised in partition_enhanced_gnns
# Recommended usage is giving the path of a metadata output of a feature vector computation as input, otherwise all other attributes must be set
# Returns the transformed dataset and 
def include_cluster_id_feature_transform(dataset: Dataset, absolute_path_prefix: str, feature_metadata_path: Optional[str], database_feature_vectors_path: Optional[str], feature_identifier: Optional[Dict[str, str]], dataset_properties_path: Optional[str], lookup_path: Optional[str], cluster_alg: Clustering_Algorithm, centroids: Optional[np.array] = None, medoids: Optional[np.array] = None) -> Tuple[Dataset, float]:

    t0 = time.time()

    # sanity checks of the input
    assert feature_metadata_path is not None or (database_feature_vectors_path is not None and feature_identifier is not None and dataset_properties_path is not None and lookup_path is not None)

    if feature_metadata_path is not None:
        # Read relevant info from metadata file

        feature_metadata = {}

        path = osp.join(absolute_path_prefix, feature_metadata_path)
        feature_metadata = util.read_metadata_file(path = path)
            
        database_feature_vectors_path = feature_metadata["result_prop"]["path"]
        feature_identifier = feature_metadata["result_prop"]["feature_identifier"]
        dataset_properties_path = feature_metadata["dataset_prop"]["properties_file_path"]
        lookup_path = feature_metadata["dataset_prop"]["idx_lookup_path"]

    num_graphs = dataset.len()

    # Read dataset properties and indexing necessary to compute feature vectors in case they have not been computed yet
    dataset_properties = Dataset_Properties_Manager(absolute_path_prefix = absolute_path_prefix, properties_path = dataset_properties_path)
    dataset_properties.initialize_idx_lookups(lookup_path = lookup_path)
    node_pred = dataset_properties.properties["node_pred"]

    # Load computed feature vectors
    path = osp.join(absolute_path_prefix, database_feature_vectors_path)
    if not osp.exists(path):
        raise FileNotFoundError
    
    feature_data, _ = load_svmlight_file(f = path, dtype='float64', zero_based = True)

    for graph_id in range(num_graphs):
        graph_data = dataset.get(graph_id)
        num_vertices = graph_data.num_nodes
        if node_pred:
            # untested
            cluster_ids = torch.zeros(num_vertices)
            if feature_identifier["id"] == "k-disk_sp":
                distances_alphabet = list(range(feature_identifier["k"]))
            elif feature_identifier["id"] == "r-s-ring_sp":
                distances_alphabet = list(range(feature_identifier["s"]))
            sp_features = SP_graph_features(label_alphabet = dataset_properties.properties["label_alphabet"], distances_alphabet = distances_alphabet)
            for vertex_id in range(num_vertices):
                cluster_ids[vertex_id] = get_cluster_id_node_pred(vertex_id = vertex_id, dataset_properties = dataset_properties, algorithm = cluster_alg, graph = graph_data, feature_vectors = feature_data, feature_identifier = feature_identifier, sp_features = sp_features, centroids = centroids, medoids = medoids).item()
            graph_data.x = torch.cat((cluster_ids, graph_data.x), dim = 1)
        else:
            cluster_ids = get_cluster_ids(graph_id = graph_id, dataset_properties = dataset_properties, algorithm = cluster_alg, graph = graph_data, feature_vectors_database = feature_data, feature_identifier = feature_identifier, centroids = centroids, medoids = medoids)
            graph_data.x = torch.cat((cluster_ids, graph_data.x), dim = 1)

    return dataset, time.time() - t0


# Returns the cluster_ids for all vertices in a given graph with id graph_id
# Currently only works with centroids or medoids based clustering algorithm
# Currently only implemented for graph_pred tasks
def get_cluster_ids(graph_id: int, dataset_properties: Dataset_Properties_Manager, algorithm: Clustering_Algorithm, graph: Optional[Data] = None, feature_vectors_database: Optional[spmatrix] = None, feature_identifier: Dict[str, str] = None, centroids: Optional[np.array] = None, medoids: Optional[np.array] = None) -> Tensor:
    
    # a list of all feature vectors of vertices in the given graph
    feature_vectors_graph = []

    if graph is None:
        # We assume the given graph to be in the dataset
        num_vertices = dataset_properties.properties["graph_sizes"][graph_id]
    else:
        num_vertices = graph.num_nodes

    if feature_vectors_database is not None and graph_id in dataset_properties.database_graph_start_idx:
        # feature vectors have already been computed

        for vertex_id in range(num_vertices):
            vertex_identifier = tuple([graph_id, vertex_id])
        
            database_idx = dataset_properties.get_database_idx_from_vertex_identifier(vertex_identifier = vertex_identifier)
            assert feature_vectors_database[database_idx,0] == graph_id and feature_vectors_database[database_idx,1] == vertex_id
            feature_vectors_graph.append(feature_vectors_database[database_idx, 2:])
    else:
        # feature vectors have to be computed
        assert graph is not None

        assert feature_identifier is not None
        if feature_identifier["id"] == "k-disk_sp":
            k = int(feature_identifier["k"])
            # k-disk sp features
            distances_alphabet = list(range(k))
            sp_features = SP_graph_features(label_alphabet = dataset_properties.properties["label_alphabet"], distances_alphabet = distances_alphabet)
            for vertex_id in range(num_vertices):
                feature, _ = compute_single_k_disk_sp_feature_vector(graph = graph, vertex_id = vertex_id, k = k, sp_features = sp_features)
                feature_vectors_graph.append(feature)

        elif feature_identifier["id"] == "r-s-ring_sp":
            r = int(feature_identifier["r"])
            s = int(feature_identifier["s"])
            # r-s-ring sp features
            distances_alphabet = list(range(s))
            sp_features = SP_graph_features(label_alphabet = dataset_properties.properties["label_alphabet"], distances_alphabet = distances_alphabet)
            for vertex_id in range(num_vertices):
                feature, _ = compute_single_r_s_ring_sp_feature_vector(graph = graph, vertex_id = vertex_id, r = r, s = s, sp_features = sp_features)
                feature_vectors_graph.append(feature)

        elif feature_identifier["id"] == "vertex_sp":
            # vertex sp features
            distances_alphabet = list(range(dataset_properties.properties["graph_size"]["max"]))
            sp_features = SP_vertex_features(label_alphabet = dataset_properties.properties["label_alphabet"], distances_alphabet = distances_alphabet)
            feature_vectors_graph, _ = compute_vertex_sp_feature_vectors(graph = graph, sp_features = sp_features)

    return get_cluster_ids_from_feature_vectors(feature_vectors = feature_vectors_graph, cluster_alg = algorithm, centroids = centroids, medoids = medoids)
    
def get_cluster_id_node_pred(vertex_id: int, dataset_properties: Dataset_Properties_Manager, algorithm: Clustering_Algorithm, graph: Optional[Data] = None, feature_vectors: Optional[spmatrix] = None, feature_identifier: Dict[str, str] = None, sp_features: Optional[SP_graph_features] = None, centroids: Optional[np.array] = None, medoids: Optional[np.array] = None) -> int:
    # untested

    res = None
    
    vertex_identifier = tuple([0, vertex_id])

    dataset_idx = dataset_properties.get_dataset_idx_from_vertex_identifier(vertex_identifier)

    if feature_vectors is not None and dataset_idx in dataset_properties.dataset_idx_lookup:
        # A feature vector has previously been computed
        database_idx = dataset_properties.get_database_idx_from_vertex_identifier(vertex_identifier = vertex_identifier)
        res = feature_vectors[database_idx, 2:]
    else:
        # A feature vector has to be newly computed

        assert feature_identifier is not None

        if feature_identifier["id"] == "k-disk_sp":
            # k-disk sp features
            k = int(feature_identifier["k"])
            res, _ = compute_single_k_disk_sp_feature_vector(graph = graph, vertex_id = vertex_id, k = k, sp_features = sp_features)

        elif feature_identifier["id"] == "r-s-ring_sp":
            # r-s-ring sp features
            r = int(feature_identifier["r"])
            s = int(feature_identifier["s"])
            res, _ = compute_single_r_s_ring_sp_feature_vector(graph = graph, vertex_id = vertex_id, r = r, s = s, sp_features = sp_features)

    return get_cluster_ids_from_feature_vectors(feature_vectors = [res], cluster_alg = algorithm, centroids = centroids, medoids = medoids)

# generate: Whether the feature vectors are newly computed.
def get_cluster_ids_from_feature_vectors(feature_vectors: List[np.array], cluster_alg: Clustering_Algorithm, centroids: Optional[np.array] = None, medoids: Optional[np.array] = None) -> Tensor:
    result = None
    
    if cluster_alg == Clustering_Algorithm.k_means:
        assert centroids is not None
        feature_vectors = np.concatenate(feature_vectors, axis = 0)
        result = torch.tensor(pairwise_distances_argmin(feature_vectors, centroids))
    elif cluster_alg == Clustering_Algorithm.clarans:
        assert medoids is not None
        feature_vectors = np.concatenate(feature_vectors, axis = 0)
        result = torch.tensor(pairwise_distances_argmin(feature_vectors, medoids))
    else:
        raise ValueError('Invalid clustering algorithm passed to get_cluster_id')
    
    return result