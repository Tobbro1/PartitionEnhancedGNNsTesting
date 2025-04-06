import os.path as osp
import numpy as np
import os

from clustering import Clustering_Algorithm, Vertex_Partition_Clustering
import util
import gnn_utils

from CSL_dataset import CSL_Dataset
import constants

def run_experiment(root_path: str, working_path: str, vertex_feature_metadata_path: str):

    vertex_feature_metadata = util.read_metadata_file(path = osp.join(root_path, vertex_feature_metadata_path))

    # create a clustering with lsa
    clusterer = Vertex_Partition_Clustering(absolute_path_prefix = root_path)
    feature_vector_database_path = vertex_feature_metadata["result_prop"]["path"]
    dataset_desc = vertex_feature_metadata["dataset_prop"]["desc"]

    # split_prop is expected to be
    # ["desc"]
    # ["split_mode"]
    # ["num_samples"]
    # ["split_idx"] if split_mode is 'CV'
    split_desc = "Test_desc"
    split_mode = "Test"
    num_samples = -2
    split_idx = -1
    split_prop = { "desc" : split_desc, "split_mode" : split_mode, "num_samples" : num_samples, "split_idx" : split_idx}
    normalize = True

    clusterer.load_dataset_from_svmlight(path = feature_vector_database_path, dtype = 'float64', dataset_desc = dataset_desc, split_prop = split_prop, normalize = normalize)
    
    # LSA
    target_dim = 2
    lsa_filename = f'{target_dim}_dim_lsa.pkl'
    clusterer.generate_lsa(target_dimensions = target_dim, write_lsa_path = working_path, write_lsa_filename = lsa_filename)

    clusterer.apply_lsa_to_dataset()

    # k-means
    mbk_n_clusters = 6
    mbk_batch_size = 1024
    mbk_n_init = 10
    mbk_max_no_improvement = 10
    mbk_max_iter = 1000

    centroids_filename = f'{mbk_n_clusters}-means_centroids.txt'

    labels, centroids, inertia, clustering_time = clusterer.mini_batch_k_means(n_clusters = mbk_n_clusters, batch_size = mbk_batch_size, n_init = mbk_n_init, max_no_improvement = mbk_max_no_improvement, max_iter = mbk_max_iter)

    clusterer.write_centroids_or_medoids(points = centroids, path = working_path, filename = centroids_filename)

    metadata_filename = 'cluster_metadata.json'
    clusterer.write_metadata(path = working_path, filename = metadata_filename)

    # TODO: Test gnn_utils.py
    csl_path = osp.join('data', 'CSL', 'CSL_dataset')
    dataset_csl = CSL_Dataset(root = osp.join(root_path, csl_path))

    feature_metadata_path = vertex_feature_metadata_path
    cluster_metadata_path = osp.join(working_path, metadata_filename)

    print(dataset_csl[0].x)

    dataset_csl, add_cluster_id_time = gnn_utils.include_cluster_id_feature_transform(dataset = dataset_csl, absolute_path_prefix = root_path, feature_metadata_path = feature_metadata_path, cluster_metadata_path = cluster_metadata_path)

    print(dataset_csl[0].x)

if __name__ == '__main__':
    # test gnn util

    constants.initialize_random_seeds()

    root_path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir)
    working_path = osp.join('data', 'CSL', 'CSL_dataset', 'results', 'vertex_sp_features')
    feature_metadata_path = osp.join(working_path, 'metadata.json')
    run_experiment(root_path = root_path, working_path = osp.join(working_path, 'cluster-gnn'), vertex_feature_metadata_path = feature_metadata_path)
