#TODO:
#Create a class that incorporates functionality to:
#Read the saved feature vectors from a file
#Cluster these feature vectors using different clustering techniques (k-means, OPTICS)
#Include functionality to write the clustering results into a file (a dictionary (graph_id, vertex_id) -> cluster_id ?)

# system
import time
import os.path as osp
import os
import pickle
import json

from copy import deepcopy

# general
from typing import Dict, Tuple, List, Optional
from enum import Enum

# pytorch
import torch
from torch import Tensor

# scikit-learn
import sklearn
from sklearn.datasets import load_svmlight_file
from sklearn.cluster import MiniBatchKMeans, OPTICS
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import TruncatedSVD
import sklearn.preprocessing as prepocessing

# numpy
import numpy as np 

# plotting
import matplotlib.pyplot as plt
import colorsys

# constants
import constants

# own functionality
from clarans import Clarans
import util

class Clustering_Algorithm(Enum):
    k_means = 0
    optics = 1
    clarans = 2


class Vertex_Partition_Clustering():

    def __init__(self, absolute_path_prefix: str):
        super().__init__()

        self.absolute_path_prefix = absolute_path_prefix

        self.split = None

        self.original_dataset = None
        self.dataset = None
        self.transformed_dataset = None
        self.vertex_identifier = None

        self.num_vertices = -1
        self.num_features = -1

        self.lsa = None
        self.draw_lsa = None

        # Metadata collected while clustering
        # dataset_prop: path, desc, normalized
        # clustering_alg: desc: short description of the clustering algorithm used
        #                 identifier: id of the clustering algorithm (from the enum)
        #                 metric: short string
        # datasplit_prop: desc, split_mode (CV/fixed), num_samples, split_idx (if CV)
        # lsa: lsa_used,
        #      path (of pickle file) (if used)
        #      result_prop: explained_variances (for each component, var and var_ratio), num_features_seen, time
        #      config (if used): num_components, algorithm (as str; arpack or randomized), num_iter (if random), num_oversamples (if random), 
        #                        power_iteration_normalizer (if randomized), random_seed (if random), tol (if arpack)
        # result_prop: desc (short string describing the result), path
        #              size of result (in bytes): as array, on disk
        #              if k_means: num_centroids, num_dim, inertia, num_iter, num_steps, num_features_seen
        # times: read_dataset_time, clustering_time
        # config: if k_means: num_clusters, init (method as string), max_iter, batch_size, random_seed, max_no_improvement, num_inits, init_size, reassignment_ratio
        self.metadata = {}
        self.metadata["path_prefix"] = self.absolute_path_prefix
        self.metadata["dataset_prop"] = {}
        self.metadata["dataset_prop"]["desc"] = ""
        self.metadata["dataset_prop"]["normalized"] = ""
        self.metadata["dataset_prop"]["path"] = ""
        self.metadata["clustering_alg"] = {}
        self.metadata["clustering_alg"]["desc"] = ""
        self.metadata["clustering_alg"]["metric"] = ""
        self.metadata["clustering_alg"]["id"] = -1
        self.metadata["data_split"] = {}
        self.metadata["data_split"]["desc"] = ""
        self.metadata["data_split"]["split_mode"] = ""
        self.metadata["data_split"]["num_samples"] = ""
        # self.metadata["data_split"]["split_idx"] = -1
        self.metadata["lsa"] = {}
        self.metadata["lsa"]["lsa_used"] = False
        # if lsa_used
        # self.metadata["lsa"]["path"] = ""
        # self.metadata["lsa"]["result_prop"] = {}
        # self.metadata["lsa"]["result_prop"]["num_features_seen"] = -1
        # self.metadata["lsa"]["result_prop"]["explained_variances"] = {}
        # # foreach component: { variance, var_ratio } 
        # self.metadata["lsa"]["config"] = {}
        # self.metadata["lsa"]["config"]["algorithm"] = ""
        # self.metadata["lsa"]["config"]["num_components"] = -1
        # # if randomized:
        # # self.metadata["lsa"]["config"]["num_iter"] = -1
        # # self.metadata["lsa"]["config"]["num_oversamples"] = -1
        # # self.metadata["lsa"]["config"]["power_iteration_normalizer"] = ""
        # # if arpack:
        # # self.metadata["lsa"]["config"]["tol"] = -1.0
        self.metadata["result_prop"] = {}
        self.metadata["result_prop"]["desc"] = ""
        self.metadata["result_prop"]["path"] = ""
        self.metadata["result_prop"]["size"] = {}
        self.metadata["result_prop"]["size"]["array"] = -1
        self.metadata["result_prop"]["size"]["disk"] = -1
        # if k_means
        # self.metadata["result_prop"]["num_centroids"] = -1
        # self.metadata["result_prop"]["num_dim"] = -1
        # self.metadata["result_prop"]["inertia"] = 0.0
        # self.metadata["result_prop"]["num_iter"] = -1
        # self.metadata["result_prop"]["num_steps"] = -1
        # self.metadata["result_prop"]["num_features_seen"] = -1
        self.metadata["times"] = {}
        self.metadata["times"]["read_from_disk"] = -1.0
        self.metadata["times"]["write_on_disk"] = -1.0
        # self.metadata["times"]["lsa_comp"] = -1.0 # if lsa
        self.metadata["times"]["clustering"] = -1.0
        self.metadata["config"] = {}
        # # if k_means
        # self.metadata["config"]["num_clusters"] = -1
        # self.metadata["config"]["init_method"] = ""
        # self.metadata["config"]["max_iter"] = -1
        # self.metadata["config"]["batch_size"] = -1
        # self.metadata["config"]["max_no_improvement"] = -1
        # self.metadata["config"]["num_inits"] = -1
        # self.metadata["config"]["num_init_random_samples"] = -1
        # self.metadata["config"]["reassignment_ratio"] = -1.0
        # self.metadata["config"]["tol"] = -1.0


    def reset_parameters_and_metadata(self) -> None:
        self.dataset = None
        self.transformed_dataset = None
        self.vertex_identifier = None

        self.num_vertices = -1
        self.num_features = -1

        self.lsa = None
        self.draw_lsa = None

        # Metadata collected while clustering
        # dataset_prop: path, desc, normalized
        # clustering_alg: desc: short description of the clustering algorithm used
        #                 identifier: id of the clustering algorithm (from the enum)
        #                 metric: short string
        # datasplit_prop: desc, split_mode (CV/fixed), num_samples, split_idx (if CV)
        # lsa: lsa_used,
        #      path (of pickle file) (if used)
        #      result_prop: explained_variances (for each component, var and var_ratio), num_features_seen, time
        #      config (if used): num_components, algorithm (as str; arpack or randomized), num_iter (if random), num_oversamples (if random), 
        #                        power_iteration_normalizer (if randomized), random_seed (if random), tol (if arpack)
        # result_prop: desc (short string describing the result), path
        #              size of result (in bytes): as array, on disk
        #              if k_means: num_centroids, num_dim, inertia, num_iter, num_steps, num_features_seen
        # times: read_dataset_time, clustering_time
        # config: if k_means: num_clusters, init (method as string), max_iter, batch_size, random_seed, max_no_improvement, num_inits, init_size, reassignment_ratio
        self.metadata = {}
        self.metadata["dataset_prop"] = {}
        self.metadata["dataset_prop"]["desc"] = ""
        self.metadata["dataset_prop"]["normalized"] = ""
        self.metadata["dataset_prop"]["path"] = ""
        self.metadata["clustering_alg"] = {}
        self.metadata["clustering_alg"]["desc"] = ""
        self.metadata["clustering_alg"]["metric"] = ""
        self.metadata["clustering_alg"]["id"] = -1
        self.metadata["data_split"] = {}
        self.metadata["data_split"]["desc"] = ""
        self.metadata["data_split"]["split_mode"] = ""
        self.metadata["data_split"]["num_samples"] = ""
        # self.metadata["data_split"]["split_idx"] = -1
        self.metadata["lsa"] = {}
        self.metadata["lsa"]["lsa_used"] = False
        # if lsa_used
        # self.metadata["lsa"]["path"] = ""
        # self.metadata["lsa"]["result_prop"] = {}
        # self.metadata["lsa"]["result_prop"]["num_features_seen"] = -1
        # self.metadata["lsa"]["result_prop"]["explained_variances"] = {}
        # # foreach component: { variance, var_ratio } 
        # self.metadata["lsa"]["config"] = {}
        # self.metadata["lsa"]["config"]["algorithm"] = ""
        # self.metadata["lsa"]["config"]["num_components"] = -1
        # # if randomized:
        # # self.metadata["lsa"]["config"]["num_iter"] = -1
        # # self.metadata["lsa"]["config"]["num_oversamples"] = -1
        # # self.metadata["lsa"]["config"]["power_iteration_normalizer"] = ""
        # # if arpack:
        # # self.metadata["lsa"]["config"]["tol"] = -1.0
        self.metadata["result_prop"] = {}
        self.metadata["result_prop"]["desc"] = ""
        self.metadata["result_prop"]["path"] = ""
        self.metadata["result_prop"]["size"] = {}
        self.metadata["result_prop"]["size"]["array"] = -1
        self.metadata["result_prop"]["size"]["disk"] = -1
        # if k_means
        # self.metadata["result_prop"]["num_centroids"] = -1
        # self.metadata["result_prop"]["num_dim"] = -1
        # self.metadata["result_prop"]["inertia"] = 0.0
        # self.metadata["result_prop"]["num_iter"] = -1
        # self.metadata["result_prop"]["num_steps"] = -1
        # self.metadata["result_prop"]["num_features_seen"] = -1
        self.metadata["times"] = {}
        self.metadata["times"]["read_from_disk"] = -1.0
        self.metadata["times"]["write_on_disk"] = -1.0
        # self.metadata["times"]["lsa_comp"] = -1.0 # if lsa
        # self.metadata["times"]["lsa_application"] = -1.0
        self.metadata["times"]["clustering"] = -1.0
        self.metadata["config"] = {}
        # # if k_means
        # self.metadata["config"]["num_clusters"] = -1
        # self.metadata["config"]["init_method"] = ""
        # self.metadata["config"]["max_iter"] = -1
        # self.metadata["config"]["batch_size"] = -1
        # self.metadata["config"]["max_no_improvement"] = -1
        # self.metadata["config"]["num_inits"] = -1
        # self.metadata["config"]["num_init_random_samples"] = -1
        # self.metadata["config"]["reassignment_ratio"] = -1.0
        # self.metadata["config"]["tol"] = -1.0

    # Returns a tuple of the learned components and the ratio of their explained variance
    def generate_lsa(self, target_dimensions: int, write_lsa_path: Optional[str] = None, write_lsa_filename: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:

        t0 = time.time()

        # log metadata for lsa
        if write_lsa_filename is not None:
            self.metadata["lsa"]["path"] = ""

        algorithm = 'arpack'
        tol = 0.0
        n_iter = 5 # The number of iterations of the randomized SVD solver; default: 5
        n_oversamples = 10 # The number of oversamples for randomized SVD solver; default: 10
        power_iteration_normalizer = 'auto' # Power iteration normalizer for randomized SVD solver; default: 'auto'

        self.metadata["lsa"]["result_prop"] = {}
        self.metadata["lsa"]["config"] = {}
        self.metadata["lsa"]["config"]["algorithm"] = algorithm
        self.metadata["lsa"]["config"]["num_components"] = target_dimensions

        if algorithm == 'randomized':
            self.metadata["lsa"]["config"]["num_iter"] = n_iter
            self.metadata["lsa"]["config"]["num_oversamples"] = n_oversamples
            self.metadata["lsa"]["config"]["power_iteration_normalizer"] = power_iteration_normalizer
        elif algorithm == 'arpack':
            self.metadata["lsa"]["config"]["tol"] = tol

        # n_components is the number of principal components that are utilised, thus the number of dimensions after the LSA
        self.lsa = TruncatedSVD(n_components = target_dimensions, algorithm = algorithm, n_iter = n_iter, n_oversamples = n_oversamples, power_iteration_normalizer = power_iteration_normalizer, tol = tol)
        self.lsa.fit(self.original_dataset)

        self.metadata["lsa"]["result_prop"]["num_features_seen"] = self.lsa.n_features_in_
        self.metadata["lsa"]["result_prop"]["explained_variances"] = {}
        sum_explained_var_ratio = 0.0
        for idx in range(self.lsa.explained_variance_.shape[0]):
            self.metadata["lsa"]["result_prop"]["explained_variances"][idx] = { "var" : self.lsa.explained_variance_[idx], "var_ratio" : self.lsa.explained_variance_ratio_[idx] }
            sum_explained_var_ratio += self.lsa.explained_variance_ratio_[idx]
        self.metadata["lsa"]["result_prop"]["explained_variances"]["total_ratio"] = sum_explained_var_ratio

        if write_lsa_path is not None:
            # Store the lsa object for potential later use
            assert write_lsa_filename is not None

            path = osp.join(self.absolute_path_prefix, write_lsa_path)
            if not osp.exists(path):
                os.makedirs(path)

            path = osp.join(path, write_lsa_filename)
            if not osp.exists(path):
                open(path, 'wb').close()

            with open(path, 'wb') as file:
                pickle.dump(obj = self.lsa, file = file)

            self.metadata["lsa"]["path"] = osp.join(write_lsa_path, write_lsa_filename)

        self.metadata["times"]["lsa_comp"] = time.time() - t0

        return self.lsa.components_, self.lsa.explained_variance_ratio_

    def apply_lsa_to_dataset(self) -> None:
        if self.dataset is None or self.num_vertices < 1:
            raise ValueError("Dataset invalid")
        
        if self.lsa is None:
            raise ValueError("Initialize LSA before applying")
        
        t0 = time.time()
        
        self.metadata["lsa"]["lsa_used"] = True

        self.dataset = self.lsa.transform(self.original_dataset[self.split,:])

        self.metadata["times"]["lsa_application"] = time.time() - t0

    # A PCA to two dimensions used for visualization
    # Returns a tuple of the learned components and the ratio of their explained variance
    def generate_draw_pca(self) -> Tuple[np.ndarray, np.ndarray]:

        self.draw_lsa = TruncatedSVD(n_components = 2, algorithm = 'arpack')
        self.draw_lsa.fit(self.dataset)

        return self.draw_lsa.components_, self.draw_lsa.explained_variance_ratio_

    # This method draws the dataset colored by their assigned cluster according to the label parameter. This makes use of a PCA to two dimensions
    # The grid_granularity parameter is used when visualizing centroid based methods. Note that the grid is made between the smallest and highest values for the PCA transformed vectors which might lead to huge memroy requirements in case of large ranges and small granularity
    # labels is used to visualize optics results
    def draw_clustering_data(self, num_figure: int, title: str, cluster_alg: Clustering_Algorithm, labels: np.array, centroids: Optional[np.array] = None, grid_granularity: Optional[int] = 2, medoids: Optional[np.array] = None) -> None:

        if self.draw_lsa is None:
            self.generate_draw_pca()
        
        reduced_data = self.draw_lsa.transform(self.dataset)

        if cluster_alg == Clustering_Algorithm.k_means:
            # k-means, code taken from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
            assert centroids is not None
            assert grid_granularity is not None

            centroids = self.draw_lsa.transform(centroids)

            h = grid_granularity

            # Plot the decision boundary. For that, we will assign a color to each
            x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
            y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            Z = pairwise_distances_argmin(np.c_[xx.ravel(), yy.ravel()], centroids)
            Z = Z.reshape(xx.shape)

            plt.figure(num_figure)
            plt.clf()
            plt.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired, aspect="auto", origin="lower",)
            plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
            
            # Plot the centroids as a white X
            plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3, color="w", zorder=10,)
            plt.title(title)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xlabel(f"Component 1: {self.draw_lsa.explained_variance_ratio_[0]}")
            plt.ylabel(f"Component 2: {self.draw_lsa.explained_variance_ratio_[1]}")
            plt.xticks(())
            plt.yticks(())

        elif cluster_alg == Clustering_Algorithm.optics:
            # OPTICS, code taken from https://scikit-learn.org/stable/auto_examples/cluster/plot_optics.html#sphx-glr-auto-examples-cluster-plot-optics-py
            assert labels is not None

            # The -1 stems from outliers being labeled as -1 while not belonging to a cluster
            num_cluster = np.unique(labels) - 1

            # Generate colors based on the number of clusters found for plotting
            hsv_tuples = [(x*1.0/num_cluster, 0.5, 0.5) for x in range(num_cluster)]
            rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)

            # Configure the plot
            plt.figure(num_figure)
            plt.clf()
            plt.title(title)

            x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
            y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            plt.xlabel(f"Component 1: {self.draw_lsa.explained_variance_ratio_[0]}")
            plt.ylabel(f"Component 2: {self.draw_lsa.explained_variance_ratio_[1]}")

            for c in range(num_cluster):
                class_data = reduced_data[labels == c]
                plt.plot(class_data[:,0], class_data[:,1], color = rgb_tuples[c], alpha = 0.3)

            plt.plot(reduced_data[labels == -1,0], reduced_data[labels == -1,1], alpha = 0.1)

        elif cluster_alg == Clustering_Algorithm.clarans:
            # clarans
            assert labels is not None
            assert medoids is not None

            num_cluster = np.unique(labels)
            medoids = self.draw_lsa.transform(medoids)

            # Generate colors based on the number of clusters found for plotting
            hsv_tuples = [(x*1.0/num_cluster, 0.5, 0.5) for x in range(num_cluster)]
            rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)

            # Configure the plot
            plt.figure(num_figure)
            plt.clf()
            plt.title(title)

            x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
            y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            plt.xlabel(f"Component 1: {self.draw_lsa.explained_variance_ratio_[0]}")
            plt.ylabel(f"Component 2: {self.draw_lsa.explained_variance_ratio_[1]}")

            for c in range(num_cluster):
                class_data = reduced_data[labels == c]
                plt.plot(class_data[:,0], class_data[:,1], color = rgb_tuples[c], alpha = 0.3)

            # Plot the medoids as a white X
            plt.scatter(medoids[:, 0], medoids[:, 1], marker="x", s=169, linewidths=3, color="w", zorder=10,)

    # Returns a tuple of assignments, centroids, inertia and execution time of the clustering
    def mini_batch_k_means(self, n_clusters: int, batch_size: int, n_init: int, max_no_improvement: int, max_iter: int, min_cluster_size: int) -> Tuple[np.ndarray, np.ndarray, float, float]:

        t0 = time.time()

        self.metadata["clustering_alg"]["desc"] = "mini-batch k-means"
        self.metadata["clustering_alg"]["metric"] = "euclidean"
        self.metadata["clustering_alg"]["id"] = Clustering_Algorithm.k_means.value

        # batch-size greater than 256 * num_cores for paralllelism on all cores
        # n_init describes the number of random initializations tried, before using the best (in terms of inertia) initialization (note that the algorithm is not run for every initialization but only for the best)
        # init_size is the number of samples used for initialization (we use the default value which should be set to 3 * n_clusters)
        # max_no_improvements denotes the maximum number of no improvement on inertia for the algorithm to terminate early
        desc = f'{n_clusters}-means'
        init_method = 'k-means++'
        init_size = 3 * batch_size
        if init_size < n_clusters:
            init_size = 3 * n_clusters
        reassignment_ratio = 0.1
        tol = 0.0

        self.metadata["result_prop"]["desc"] = desc
        self.metadata["config"]["num_clusters"] = n_clusters
        self.metadata["config"]["init_method"] = init_method
        self.metadata["config"]["max_iter"] = max_iter
        self.metadata["config"]["min_cluster_size"] = min_cluster_size
        self.metadata["config"]["batch_size"] = batch_size
        self.metadata["config"]["max_no_improvement"] = max_no_improvement
        self.metadata["config"]["num_inits"] = n_init
        self.metadata["config"]["init_size"] = init_size
        self.metadata["config"]["reassignment_ratio"] = reassignment_ratio
        self.metadata["config"]["tol"] = tol
        
        mbk = MiniBatchKMeans(n_clusters = n_clusters, init = init_method, max_iter = max_iter, batch_size = batch_size, tol = tol, max_no_improvement = max_no_improvement, init_size = init_size, n_init = n_init, reassignment_ratio = reassignment_ratio)

        mbk.fit(self.dataset)
        labels = pairwise_distances_argmin(self.dataset, mbk.cluster_centers_)

        centroids = mbk.cluster_centers_

        num_dim = centroids.shape[1]
        num_iter = mbk.n_iter_
        num_steps = mbk.n_steps_
        num_features_seen = mbk.n_features_in_

        # Check for minimum cluster size, if a cluster has too few labels assigned, merge this cluster with the closest other cluster 
        # (by removing the centroid, then reassigning all vertices to check whether more clusters need to be merged). Start by checking the smallest cluster. This is not necessarily the best approach to merging
        finished = False
        num_clusters_merged = 0
        while not finished:
            finished = True
            smallest_cluster = 0
            smallest_cluster_size = float('inf')
            for c in range(centroids.shape[0]):
                size_cluster = labels[labels == c].shape[0]
                if size_cluster < smallest_cluster_size:
                    smallest_cluster = c
                    smallest_cluster_size = size_cluster
            
            # Smallest cluster has been found
            if smallest_cluster_size < min_cluster_size:
                # Remove the corresponding centroid
                centroids = np.delete(centroids, smallest_cluster, axis = 0)
                # reassign all labels
                labels = pairwise_distances_argmin(self.dataset, centroids)
                num_clusters_merged += 1
                finished = False
        
        if num_clusters_merged > 0:
            # We have to recompute the inertia (sum of squared distances between each datapoint and its cluster) of the clusters in case we have merged clusters
            inertia = 0.0
            for c in range(centroids.shape[0]):
                inertia += np.sum(euclidean_distances(X = self.dataset[(labels == c),:], Y = centroids[c,:].reshape(1,-1), squared = True), axis = 0, dtype = np.float64).item()
        else:
            inertia = mbk.inertia_


        self.metadata["result_prop"]["num_centroids"] = centroids.shape[0]
        self.metadata["result_prop"]["num_dim"] = num_dim
        self.metadata["result_prop"]["inertia"] = inertia
        self.metadata["result_prop"]["num_iter"] = num_iter
        self.metadata["result_prop"]["num_steps"] = num_steps
        self.metadata["result_prop"]["num_clusters_merged"] = num_clusters_merged
        self.metadata["result_prop"]["num_features_seen"] = num_features_seen

        execution_time = time.time() - t0

        self.metadata["times"]["clustering"] = execution_time
        
        return labels, centroids, inertia, execution_time

    # Returns a tuple of assignments and execution time of the clustering
    def optics(self, min_samples: int, max_eps: float = np.inf, n_jobs: int = None) -> Tuple[np.array, float]:
        # min_samples defines the number of points in a neighborhood of a point necessary for the point to be considered a core point
        # max_eps is the maximum possible distance between to points to be considered neighbors. OPTICS searches different ranges up to max_eps, defaul value is np.inf
        # n_jobs is the number of processes to run the neighbor search. None means 1 process, -1 means using all processors

        metric = 'euclidean'

        # The method used to extract clusters from reachability and ordering
        # xi determines the steepness on the reachability plot that defines a cluster boundary. Value is a float between 0 and 1
        # predecessor correction is an improvement based on http://star.informatik.rwth-aachen.de/Publications/CEUR-WS/Vol-2191/paper37.pdf
        cluster_method = 'xi'
        xi = 0.05
        predecessor_correction = True
        
        # The minimum number of points in a cluster. If None it is equal to min_samples
        minimum_cluster_size = None

        # Algorithm used for k-NN
        algorithm = 'auto'
        
        clustering = OPTICS(min_samples = min_samples, max_eps = max_eps, n_jobs = n_jobs, metric = metric, cluster_method = cluster_method, xi = xi, predecessor_correction = predecessor_correction, min_cluster_size = minimum_cluster_size, algorithm = algorithm)
        
        t0 = time.time()
        clustering.fit(self.dataset)
        labels = clustering.labels_[clustering.ordering_]
        execution_time = time.time() - t0

        return labels, execution_time
    
    # Returns a tuple of labels (shape = (num_samples,)), medoids (shape = (num_clusters, num_features)), inertia of the computed labels and execution time
    def clarans(self, num_local: int, max_neighbor: int, num_clusters: int) -> Tuple[np.array, np.array, float, float]:
        cns = Clarans(num_local = num_local, max_neighbor = max_neighbor, num_clusters = num_clusters)

        t0 = time.time()
        cns.fit(self.dataset.toarray())
        labels = cns.cluster_labels
        execution_time = time.time() - t0

        return labels, cns.data[cns.medoids,:], cns.inertia, execution_time

    # Reads an svmlight file
    # split_prop is expected to be
    # ["desc"]
    # ["split_mode"]
    # ["num_samples"]
    # ["split_idx"] if split_mode is 'CV'
    def load_dataset_from_svmlight(self, path: str, dtype: np.dtype, normalize: bool = False, split: Optional[Tensor] | Optional[np.array] | Optional[List[int]] = None, dataset_desc: Optional[str] = None, split_prop: Dict[str, str | int] = None) -> None:

        t0 = time.time()

        data_path = osp.join(self.absolute_path_prefix, path)
        if not osp.exists(data_path):
            raise FileNotFoundError

        if dataset_desc is None:
            dataset_desc = ""

        self.metadata["dataset_prop"]["desc"] = dataset_desc
        self.metadata["dataset_prop"]["normalized"] = normalize
        self.metadata["dataset_prop"]["path"] = path

        # We can discard the target labels since clustering is unsupervised
        data, _ = load_svmlight_file(f = data_path, dtype = dtype, zero_based = True)

        if split is None:
            split = np.array(list(range(data.shape[0])))

        if isinstance(split, Tensor):
            split = split.numpy()

        self.split = split

        if split_prop is not None:
            self.metadata["data_split"] = split_prop

        if normalize:
            self.original_dataset = prepocessing.normalize(data[:,2:], axis = 1)
        else:
            self.original_dataset = data[:,2:]
        self.dataset = self.original_dataset[self.split,:].copy()
        self.vertex_identifier = data[self.split,0:2]
        self.num_vertices, self.num_features = self.dataset.shape

        self.metadata["times"]["read_from_disk"] = time.time() - t0

    def set_split(self, split: Tensor, split_prop: Dict) -> None:
        self.split = split.numpy()
        self.dataset = self.original_dataset[self.split,:].copy()
        self.metadata["data_split"] = split_prop
        self.num_vertices, self.num_features = self.dataset.shape

    # Equivalent to executing generate_clustering_result_array_from_dict with the result of generate_clustering_result_dict but without the intermediary step
    def generate_clustering_result_array(self, labels) -> np.array:
        result = np.zeros(shape = (self.num_vertices, 3), dtype = int)

        for v in range(self.num_vertices):
            graph_id, vertex_id = int(self.vertex_identifier[v,0]), int(self.vertex_identifier[v,1])
            result[v,:] = np.array([graph_id, vertex_id, int(labels[v])])

        return result

    # Returns a dictionary mapping (graph_id, vertex_id) to its corresponding cluster_id
    def generate_clustering_result_dict(self, labels) -> Dict[Tuple[int, int], int]:

        result = {}

        for i in range(self.num_vertices):
            _t = tuple([int(self.vertex_identifier[i,0]), int(self.vertex_identifier[i,1])])
            result[_t] = int(labels[i])

        return result
    
    # Returns a 2D array with rows in the form (graph_id, vertex_id, cluster_id)
    def generate_clustering_result_array_from_dict(self, dict: Dict[Tuple[int, int], int]) -> np.array:
        res = np.zeros(shape = (self.num_vertices, 3), dtype = int)
        
        for idx, _t in enumerate(dict):
            res[idx, :] = np.array([_t[0], _t[1], dict[_t]])

        return res

    # Returns metadata dictionary
    def write_centroids_or_medoids(self, points: np.array, path: str, filename: str, comment: Optional[str] = None):

        t0 = time.time()

        if comment is None:
            comment = ""

        data_path = osp.join(self.absolute_path_prefix, path)
        if not osp.exists(data_path):
            os.makedirs(data_path)

        data_path = osp.join(data_path, filename)
        if not osp.exists(data_path):
            open(data_path, 'w').close()

        self.metadata["result_prop"]["path"] = osp.join(path, filename)
        self.metadata["result_prop"]["size"]["array"] = points.nbytes

        with open(data_path, "w") as file:
            np.savetxt(fname = file, X = points, comments = '#', header = comment)

        self.metadata["result_prop"]["size"]["disk"] = os.stat(path = data_path).st_size

        self.metadata["times"]["write_on_disk"] = time.time() - t0

        return self.metadata
    
    def write_metadata(self, path: str, filename: str):
        path = osp.join(self.absolute_path_prefix, path)
        util.write_metadata_file(path = path, filename = filename, data = self.metadata)

    # Write array of given labels into a file for storage (this is done in a human-readable/non-binary way for convenience)
    # NOTE: labels has to be a 1D or 2D array
    def write_clustering_labels(self, labels: np.array, path: str, comment: str) -> None:
        path = osp.join(self.absolute_path_prefix, path)
        np.savetxt(fname = path, X = labels, comments = '#', fmt = '%d', header = comment)

    def get_metadata(self) -> Dict:
        return deepcopy(self.metadata)

#Test zone
def cluster_molhiv(cluster_algs: List[Clustering_Algorithm], draw_res: bool, write_res: bool) -> None:
    clusterer = Vertex_Partition_Clustering()

    path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir, 'data', 'OGB')
    molhiv_path = osp.join(path, 'MOL_HIV')

    molhiv_vertex_sp_features_filename = "Vertex_SP_features_MOLHIV.svmlight"
    dataset_path = osp.join(molhiv_path, molhiv_vertex_sp_features_filename)

    clusterer.load_dataset_from_svmlight(dataset_path, dtype='float64')

    num_figure = 0

    if Clustering_Algorithm.k_means in cluster_algs:
        mbk_n_clusters = 6
        mbk_batch_size = 1024
        mbk_n_init = 10
        mbk_max_no_improvement = 10
        mbk_max_iter = 1000

        cluster_labels_title = f'Vertex_SP_features_{mbk_n_clusters}_means'
        cluster_labels_filename = f'{cluster_labels_title}_labels.txt'
        clusterlabels_path = osp.join(molhiv_path, cluster_labels_filename)

        labels, centroids, inertia, clustering_time = clusterer.mini_batch_k_means(n_clusters = mbk_n_clusters, batch_size = mbk_batch_size, n_init = mbk_n_init, max_no_improvement = mbk_max_no_improvement, max_iter = mbk_max_iter)

        if draw_res:
            clusterer.draw_clustering_data(num_figure = num_figure, title = cluster_labels_filename, cluster_alg = Clustering_Algorithm.k_means, centroids = centroids, grid_granularity = 0.2)
            num_figure += 1

        if write_res:
            labels_array = clusterer.generate_clustering_result_array(labels = labels)
            clusterer.write_clustering_labels(labels = labels_array, path = clusterlabels_path, comment = cluster_labels_title)

        print(f"{cluster_labels_title}:\nInertia: {inertia}, clustering time: {clustering_time}")
    
    if Clustering_Algorithm.optics in cluster_algs:
        min_samples = 5
        max_eps = 10
        n_jobs = -2

        cluster_labels_title = f'Vertex_SP_features_optics'
        cluster_labels_filename = f'{cluster_labels_title}_labels.txt'
        clusterlabels_path = osp.join(molhiv_path, cluster_labels_filename)

        labels, clustering_time = clusterer.optics(min_samples = min_samples, max_eps = max_eps, n_jobs = n_jobs)

        if draw_res:
            clusterer.draw_clustering_data(num_figure = num_figure, title = cluster_labels_filename, cluster_alg = Clustering_Algorithm.optics, labels = labels)
            num_figure += 1

        if write_res:
            labels_array = clusterer.generate_clustering_result_array(labels = labels)
            clusterer.write_clustering_labels(labels = labels_array, path = clusterlabels_path, comment = cluster_labels_title)

        print(f"{cluster_labels_title}:\nClustering time: {clustering_time}")

    if Clustering_Algorithm.clarans in cluster_algs:
        cns_n_clusters = 6
        cns_max_neighbor = 10
        cns_num_local = 10

        cluster_labels_title = f'Vertex_SP_features_clarans'
        cluster_labels_filename = f'{cluster_labels_title}_labels.txt'
        clusterlabels_path = osp.join(molhiv_path, cluster_labels_filename)

        labels, medoids, inertia, clustering_time = clusterer.clarans(num_local = cns_num_local, max_neighbor = cns_max_neighbor, num_clusters = cns_n_clusters)

        if draw_res:
            clusterer.draw_clustering_data(num_figure = num_figure, title = cluster_labels_filename, cluster_alg = Clustering_Algorithm.clarans, labels = labels, medoids = medoids)
            num_figure += 1

        if write_res:
            labels_array = clusterer.generate_clustering_result_array(labels = labels)
            clusterer.write_clustering_labels(labels = labels_array, path = clusterlabels_path, comment = cluster_labels_title)

        print(f"{cluster_labels_title}:\nInertia: {inertia}, clustering time: {clustering_time}")

    plt.show()

if __name__ == '__main__':

    #reproducability
    constants.initialize_random_seeds()

    cluster_molhiv(cluster_algs = [Clustering_Algorithm.optics], draw_res = True, write_res = True)

    # clusterer = Vertex_Partition_Clustering()

    # path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'TU')
    # mutag_path = osp.join(path, "MUTAG")
    # dd_path = osp.join(path, "DD")
    # mutag_dataset_filename = "k_disk_SP_features_MUTAG.svmlight"
    # dd_dataset_filename = "k_disk_SP_features_DD.svmlight"
    # dataset_path = osp.join(path, mutag_path, mutag_dataset_filename)
    # clusterlabels_path = osp.join(path, mutag_path, 'cluster_labels.txt')

    # clusterer.load_dataset_from_svmlight(dataset_path, dtype='float64')

    # mbk_n_clusters = 10
    # mbk_batch_size = 1024
    # mbk_n_init = 10
    # mbk_max_no_improvement = 10
    # mbk_max_iter = 1000

    # labels, centroids, inertia, clustering_time = clusterer.mini_batch_k_means(n_clusters = mbk_n_clusters, batch_size = mbk_batch_size, n_init = mbk_n_init, max_no_improvement = mbk_max_no_improvement, max_iter = mbk_max_iter)
    # clusterer.write_clustering_labels(labels = labels, path = clusterlabels_path, comment = 'Test')
    # #Computed labels: {labels}\nComputed centroids: {centroids}\n
    
    # res = clusterer.generate_clustering_result_dict(labels)
    # print(f"Inertia: {inertia}, clustering time: {clustering_time}")

    # #draw the clustering results using a 2-dimensional PCA
    # clusterer.draw_clustering_data(num_figure = 0, title = 'Testdrawing', cluster_alg = Clustering_Alg.k_means, centroids = centroids, grid_granularity = 0.2)

    # plt.show()