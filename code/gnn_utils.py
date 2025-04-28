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
import sklearn.preprocessing as prepocessing

# own functionality
from clustering import Clustering_Algorithm
from dataset_property_util import Dataset_Properties_Manager
from k_disk_sp_feature_generator import compute_single_k_disk_sp_feature_vector
from r_s_ring_sp_feature_generator import compute_single_r_s_ring_sp_feature_vector
from vertex_sp_feature_generator import compute_vertex_sp_feature_vectors
from SP_features import SP_graph_features, SP_vertex_features
import util

# Implements functionality utilised in partition enhanced gnns

# Simpler variation of the initial implementation. Clears dataset cache but might be faster due to significantly more streamlined operations
def include_cluster_id_feature_transform(dataset: Dataset, absolute_path_prefix: str, centroids: Optional[np.ndarray] = None, cluster_metadata: Optional[Dict] = None, cluster_metadata_path: Optional[str] = None, feature_dataset = None, feature_dataset_path: Optional[str] = None, vertex_feature_metadata: Optional[Dict] = None, vertex_feature_metadata_path: Optional[str] = None) -> Tuple[Dataset, float]:
    t0 = time.time()

    assert cluster_metadata is not None or cluster_metadata_path is not None
    assert feature_dataset is not None or feature_dataset_path is not None or vertex_feature_metadata is not None or vertex_feature_metadata_path is not None

    if cluster_metadata is None:
        cluster_metadata = util.read_metadata_file(path = osp.join(absolute_path_prefix, cluster_metadata_path))
    if feature_dataset is None and feature_dataset_path is None and vertex_feature_metadata is None:
        vertex_feature_metadata = util.read_metadata_file(path = osp.join(absolute_path_prefix, vertex_feature_metadata_path))

    # generate labels

    cluster_alg = Clustering_Algorithm(cluster_metadata["clustering_alg"]["id"])

    if cluster_alg == Clustering_Algorithm.k_means:
        if centroids is None:
            # read centroids
            centroids = util.read_numpy_txt(path = osp.join(absolute_path_prefix, cluster_metadata["result_prop"]["path"]))

            normalize = cluster_metadata["dataset_prop"]["normalized"]

            if feature_dataset is None:
                if feature_dataset_path is None:
                    feature_dataset_path = vertex_feature_metadata["result_prop"]["path"]

                # Read feature dataset from disk
                feature_dataset = load_svmlight_file(f = feature_dataset_path, dtype='float64', zero_based = True)[0][:,2:]
                # feature_dataset = np.delete(feature_dataset, [0,1], axis = 1)

                if normalize:
                    feature_dataset = prepocessing.normalize(feature_dataset, axis = 1)
            else:
                # If the features are passed, we have to copy them in order to ensure that we do not change the original dataset
                feature_dataset = feature_dataset.copy()

            # apply pca if necessary
            pca = None
            if cluster_metadata["pca"]["pca_used"]:
                pca = util.read_pickle(path = osp.join(absolute_path_prefix, cluster_metadata["pca"]["path"]))

                feature_dataset = pca.transform(feature_dataset)

            # get cluster_ids

            # Determine whether the centroids have 1D features or whether the centroids have only one element
            if feature_dataset.shape[1] == 1:
                # 1D features
                if np.ndim(centroids) == 0:
                    # 1 centroid
                    centroids = centroids.reshape(1,1)
                else:
                    # more than 1 centroid
                    centroids = centroids.reshape(-1,1)
            else:
                # more than 1D features
                if len(centroids.shape) == 1:
                    # Only one centroid
                    centroids = centroids.reshape(1,-1)

            cluster_ids = torch.tensor(pairwise_distances_argmin(feature_dataset, centroids))
            
            # Add the cluster_ids to the finished dataset
            dataset._data.x = torch.cat((cluster_ids.view(-1,1), dataset._data.x.view(-1, dataset.num_features)), dim = 1)

            # Delete cache since iterating over the graphs is too time consuming and the cache is rebuilt in the first training iteration
            dataset._data_list = None

    else:
        raise ValueError('Invalid cluster algorithm')
    
    return dataset, time.time() - t0


# # Transforms the features of the given dataset to include the cluster_id so they can be utilised in partition_enhanced_gnns
# # Recommended usage is giving the path of a metadata output of a feature vector computation as input, otherwise all other attributes must be set
# # Returns the transformed dataset and 
# def include_cluster_id_feature_transform(dataset: Dataset, absolute_path_prefix: str, feature_metadata_path: Optional[str] = None, feature_metadata: Optional[Dict] = None, cluster_metadata_path: Optional[str] = None, cluster_metadata: Optional[Dict] = None) -> Tuple[Dataset, float]:

#     # cluster_alg: Clustering_Algorithm, centroids: Optional[np.array] = None, medoids: Optional[np.array] = None
#     # , database_feature_vectors_path: Optional[str], feature_identifier: Optional[Dict[str, str]], dataset_properties_path: Optional[str], lookup_path: Optional[str]
#     t0 = time.time()

#     # sanity checks of the input
#     assert feature_metadata_path is not None or feature_metadata is not None
#     assert cluster_metadata_path is not None or cluster_metadata is not None

#     if feature_metadata is None:
#         feature_metadata = util.read_metadata_file(path = osp.join(absolute_path_prefix, feature_metadata_path))

#     if cluster_metadata is None:
#         cluster_metadata = util.read_metadata_file(path = osp.join(absolute_path_prefix, cluster_metadata_path))

#     centroids = None
#     medoids = None

#     cluster_alg = Clustering_Algorithm(cluster_metadata["clustering_alg"]["id"])
#     if cluster_alg == Clustering_Algorithm.k_means:
#         centroids = util.read_numpy_txt(path = osp.join(absolute_path_prefix, cluster_metadata["result_prop"]["path"]))
#     elif cluster_alg == Clustering_Algorithm.clarans:
#         medoids = util.read_numpy_txt(path = osp.join(absolute_path_prefix, cluster_metadata["result_prop"]["path"]))

#     normalize = cluster_metadata["dataset_prop"]["normalized"]

#     pca = None
#     if cluster_metadata["pca"]["pca_used"]:
#         pca = util.read_pickle(path = osp.join(absolute_path_prefix, cluster_metadata["pca"]["path"]))

#     # Read relevant dataset feature info from metadata file      
#     database_feature_vectors_path = feature_metadata["result_prop"]["path"]
#     feature_identifier = feature_metadata["result_prop"]["feature_identifier"]
#     dataset_properties_path = feature_metadata["dataset_prop"]["properties_file_path"]
#     lookup_path = feature_metadata["dataset_prop"]["idx_lookup_path"]

#     num_graphs = dataset.len()

#     # Read dataset properties and indexing necessary to compute feature vectors in case they have not been computed yet
#     dataset_properties = Dataset_Properties_Manager(absolute_path_prefix = absolute_path_prefix, properties_path = dataset_properties_path)
#     dataset_properties.initialize_idx_lookups(lookup_path = lookup_path)
#     node_pred = dataset_properties.properties["node_pred"]

#     # Load computed feature vectors
#     path = osp.join(absolute_path_prefix, database_feature_vectors_path)
#     if not osp.exists(path):
#         raise FileNotFoundError
    
#     feature_data, _ = load_svmlight_file(f = path, dtype='float64', zero_based = True)
#     start_idx = 0

#     # The dataset access via the _data attribute is strongly discouraged since it is already cached. We do it anyway since accessing via pre-transforms is not really feasible in-computation.
#     # It might be worth trying to access via a https://pytorch-geometric.readthedocs.io/en/stable/modules/transforms.html transform but accessing the relevant information quickly might be challenging
#     num_total_vertices = dataset._data.x.shape[0]
#     total_clustering_id_vector = torch.zeros(num_total_vertices)
#     for graph_id in range(num_graphs):
#         graph_data = dataset.get(graph_id)
#         num_vertices = graph_data.num_nodes
#         if node_pred:
#             pass
#         #     # untested
#         #     cluster_ids = torch.zeros(num_vertices)
#         #     if feature_identifier["id"] == "k-disk_sp":
#         #         distances_alphabet = list(range(feature_identifier["k"]))
#         #     elif feature_identifier["id"] == "r-s-ring_sp":
#         #         distances_alphabet = list(range(feature_identifier["s"]))
#         #     sp_features = SP_graph_features(label_alphabet = dataset_properties.properties["label_alphabet"], distances_alphabet = distances_alphabet)
#         #     for vertex_id in range(num_vertices):
#         #         cluster_ids[vertex_id] = get_cluster_id_node_pred(vertex_id = vertex_id, dataset_properties = dataset_properties, algorithm = cluster_alg, graph = graph_data, feature_vectors = feature_data, feature_identifier = feature_identifier, sp_features = sp_features, centroids = centroids, medoids = medoids).item()
#         #     graph_data.x = torch.cat((cluster_ids, graph_data.x), dim = 1)
#         else:
#             cluster_ids = get_cluster_ids(graph_id = graph_id, dataset_properties = dataset_properties, algorithm = cluster_alg, pca = pca, normalize = normalize, graph = graph_data, feature_vectors_database = feature_data, feature_identifier = feature_identifier, centroids = centroids, medoids = medoids)
#             # cluster_ids = cluster_ids.view(-1,1)
#             total_clustering_id_vector[start_idx:start_idx + num_vertices] = cluster_ids
#             cluster_ids = cluster_ids.view(-1,1)
#             dataset._data_list[graph_id].x = torch.cat((cluster_ids, dataset._data_list[graph_id].x.view(-1,1)), dim = 1)
#         start_idx += num_vertices

#     # This access is again strongly discouraged and bad practice.
#     if dataset._data.x.dim() == 1:
#          dataset._data.x = dataset._data.x.view(-1,1)
#     dataset._data.x = torch.cat((total_clustering_id_vector.view(-1,1), dataset._data.x.view(-1,1)), dim = 1)

#     return dataset, time.time() - t0

def transform_feature_vector(vector, normalize: bool, pca = None) -> np.ndarray:
    if not isinstance(vector, np.ndarray):
        vector = vector.toarray()
    
    if normalize:
        vector = prepocessing.normalize(vector, axis = 1)

    if pca is not None:
        vector = pca.transform(vector)

    return vector

# Returns the cluster_ids for all vertices in a given graph with id graph_id
# Currently only works with centroids or medoids based clustering algorithm
# Currently only implemented for graph_pred tasks
def get_cluster_ids(graph_id: int, dataset_properties: Dataset_Properties_Manager, algorithm: Clustering_Algorithm, pca, normalize: bool = False, graph: Optional[Data] = None, feature_vectors_database: Optional[spmatrix] = None, feature_identifier: Dict[str, str] = None, centroids: Optional[np.ndarray] = None, medoids: Optional[np.ndarray] = None) -> Tensor:
    
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
            vector = feature_vectors_database[database_idx, 2:]
            vector = transform_feature_vector(vector = vector, normalize = normalize, pca = pca)
            feature_vectors_graph.append(vector)
    else:
        # feature vectors have to be computed
        # untested
        assert graph is not None

        assert feature_identifier is not None
        if feature_identifier["id"] == "k-disk_sp":
            k = int(feature_identifier["k"])
            # k-disk sp features
            distances_alphabet = list(range(k))
            sp_features = SP_graph_features(label_alphabet = dataset_properties.properties["label_alphabet"], distances_alphabet = distances_alphabet)
            for vertex_id in range(num_vertices):
                feature, _ = compute_single_k_disk_sp_feature_vector(graph = graph, vertex_id = vertex_id, k = k, sp_features = sp_features)
                feature = transform_feature_vector(vector = feature, normalize = normalize, pca = pca)
                feature_vectors_graph.append(feature)

        elif feature_identifier["id"] == "r-s-ring_sp":
            r = int(feature_identifier["r"])
            s = int(feature_identifier["s"])
            # r-s-ring sp features
            distances_alphabet = list(range(s))
            sp_features = SP_graph_features(label_alphabet = dataset_properties.properties["label_alphabet"], distances_alphabet = distances_alphabet)
            for vertex_id in range(num_vertices):
                feature, _ = compute_single_r_s_ring_sp_feature_vector(graph = graph, vertex_id = vertex_id, r = r, s = s, sp_features = sp_features)
                feature = transform_feature_vector(vector = feature, normalize = normalize, pca = pca)
                feature_vectors_graph.append(feature)

        elif feature_identifier["id"] == "vertex_sp":
            # vertex sp features
            distances_alphabet = list(range(dataset_properties.properties["graph_size"]["max"]))
            sp_features = SP_vertex_features(label_alphabet = dataset_properties.properties["label_alphabet"], distances_alphabet = distances_alphabet)
            feature_vectors_graph, _ = compute_vertex_sp_feature_vectors(graph = graph, sp_features = sp_features)
            for feature in feature_vectors_graph:
                feature = transform_feature_vector(vector = feature, normalize = normalize, pca = pca)
            
    return get_cluster_ids_from_feature_vectors(feature_vectors = feature_vectors_graph, cluster_alg = algorithm, centroids = centroids, medoids = medoids)
    
# def get_cluster_id_node_pred(vertex_id: int, dataset_properties: Dataset_Properties_Manager, algorithm: Clustering_Algorithm, graph: Optional[Data] = None, feature_vectors: Optional[spmatrix] = None, feature_identifier: Dict[str, str] = None, sp_features: Optional[SP_graph_features] = None, centroids: Optional[np.array] = None, medoids: Optional[np.array] = None) -> int:
#     # untested

#     res = None
    
#     vertex_identifier = tuple([0, vertex_id])

#     dataset_idx = dataset_properties.get_dataset_idx_from_vertex_identifier(vertex_identifier)

#     if feature_vectors is not None and dataset_idx in dataset_properties.dataset_idx_lookup:
#         # A feature vector has previously been computed
#         database_idx = dataset_properties.get_database_idx_from_vertex_identifier(vertex_identifier = vertex_identifier)
#         res = feature_vectors[database_idx, 2:]
#     else:
#         # A feature vector has to be newly computed

#         assert feature_identifier is not None

#         if feature_identifier["id"] == "k-disk_sp":
#             # k-disk sp features
#             k = int(feature_identifier["k"])
#             res, _ = compute_single_k_disk_sp_feature_vector(graph = graph, vertex_id = vertex_id, k = k, sp_features = sp_features)

#         elif feature_identifier["id"] == "r-s-ring_sp":
#             # r-s-ring sp features
#             r = int(feature_identifier["r"])
#             s = int(feature_identifier["s"])
#             res, _ = compute_single_r_s_ring_sp_feature_vector(graph = graph, vertex_id = vertex_id, r = r, s = s, sp_features = sp_features)

#     return get_cluster_ids_from_feature_vectors(feature_vectors = [res], cluster_alg = algorithm, centroids = centroids, medoids = medoids)

# generate: Whether the feature vectors are newly computed.
def get_cluster_ids_from_feature_vectors(feature_vectors: List[np.ndarray], cluster_alg: Clustering_Algorithm, centroids: Optional[np.ndarray] = None, medoids: Optional[np.ndarray] = None) -> Tensor:
    result = None
    
    if cluster_alg == Clustering_Algorithm.k_means:
        assert centroids is not None
        feature_vectors = np.vstack(feature_vectors)
        # Determine whether the centroids have 1D features or whether the centroids have only one element
        if feature_vectors.shape[1] == 1:
            # 1D features
            if np.ndim(centroids) == 0:
                # 1 centroid
                centroids = centroids.reshape(1,1)
            else:
                # more than 1 centroid
                centroids = centroids.reshape(-1,1)
        else:
            # more than 1D features
            if len(centroids.shape) == 1:
                # Only one centroid
                centroids = centroids.reshape(1,-1)

        result = torch.tensor(pairwise_distances_argmin(feature_vectors, centroids))
    elif cluster_alg == Clustering_Algorithm.clarans:
        assert medoids is not None
        feature_vectors = np.vstack(feature_vectors)
        result = torch.tensor(pairwise_distances_argmin(feature_vectors, medoids))
    else:
        raise ValueError('Invalid clustering algorithm passed to get_cluster_id')
    
    return result