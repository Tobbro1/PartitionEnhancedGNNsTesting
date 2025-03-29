#TODO:
#Create a class that incorporates functionality to:
#Read the saved feature vectors from a file
#Cluster these feature vectors using different clustering techniques (k-means, OPTICS)
#Include functionality to write the clustering results into a file (a dictionary (graph_id, vertex_id) -> cluster_id ?)

#system
import time
import os.path as osp

#general
from typing import Dict, Tuple, List
from enum import Enum

#scikit-learn
import sklearn
from sklearn.datasets import load_svmlight_file
from sklearn.cluster import MiniBatchKMeans, OPTICS
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.decomposition import TruncatedSVD

#numpy
import numpy as np 

#plotting
import matplotlib.pyplot as plt
import colorsys

# constants
import constants

class Clustering_Algorithm(Enum):
    k_means = 0
    optics = 1


class Vertex_Partition_Clustering():

    def __init__(self):
        super().__init__()

        self.dataset = None
        self.transformed_dataset = None
        self.vertex_identifier = None

        self.num_vertices = -1
        self.num_features = -1

        self.lsa = None
        self.draw_lsa = None

    # Returns a tuple of the learned components and the ratio of their explained variance
    def generate_lsa(self, target_dimensions) -> Tuple[np.ndarray, np.ndarray]:

        # n_components is the number of principal components that are utilised, thus the number of dimensions after the PCA
        # if copy is set to False, the data passed to train the PCA gets overwritten
        # whiten can sometimes improve downstream accuracy of estimators but removes some information from the transformed signal as it normalizes component wise variances
        # svd solver selects the method to solve the underlying singular value decomposition (see PCA documentation for values)
        self.lsa = TruncatedSVD(n_components = target_dimensions, algorithm = 'arpack')
        self.lsa.fit(self.dataset)

        return self.lsa.components_, self.lsa.explained_variance_ratio_

    def apply_lsa_to_dataset(self) -> None:
        if self.dataset is None or self.num_vertices < 1:
            raise ValueError("Dataset invalid")
        
        if self.lsa is None:
            raise ValueError("Initialize LSA before applying")

        self.dataset = self.lsa.transform(self.dataset)

    # A PCA to two dimensions used for visualization
    # Returns a tuple of the learned components and the ratio of their explained variance
    def generate_draw_pca(self) -> Tuple[np.ndarray, np.ndarray]:

        self.draw_lsa = TruncatedSVD(n_components = 2, algorithm = 'arpack')
        self.draw_lsa.fit(self.dataset)

        return self.draw_lsa.components_, self.draw_lsa.explained_variance_ratio_

    # This method draws the dataset colored by their assigned cluster according to the label parameter. This makes use of a PCA to two dimensions
    # The grid_granularity parameter is used when visualizing centroid based methods. Note that the grid is made between the smallest and highest values for the PCA transformed vectors which might lead to huge memroy requirements in case of large ranges and small granularity
    # labels is used to visualize optics results
    def draw_clustering_data(self, num_figure: int, title: str, cluster_alg: Clustering_Algorithm, centroids: np.array = None, grid_granularity = 2, labels: np.array = None) -> None:

        if self.draw_lsa is None:
            self.generate_draw_pca()
        
        reduced_data = self.draw_lsa.transform(self.dataset)

        if cluster_alg == Clustering_Algorithm.k_means:
            # k-means, code taken from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
            assert centroids is not None

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

            plt.xlabel(f"X")
            plt.ylabel(f"Y")

            for c in range(num_cluster):
                class_data = reduced_data[labels == c]
                plt.plot(class_data[:,0], class_data[:,1], color = rgb_tuples[c], alpha = 0.3)

            plt.plot(reduced_data[labels == -1,0], reduced_data[labels == -1,1], alpha = 0.1)


    # Returns a tuple of assignments, centroids, inertia and execution time of the clustering
    def mini_batch_k_means(self, n_clusters: int, batch_size: int, n_init: int, max_no_improvement: int, max_iter: int) -> Tuple[np.ndarray, np.ndarray, float, float]:

        # batch-size greater than 256 * num_cores for paralllelism on all cores
        # Note that the random_state value is not set, thus inherited from the global numpy random_state value
        # n_init describes the number of random initializations tried, before using the best (in terms of inertia) initialization (note that the algorithm is not run for every initialization but only for the best)
        # init_size is the number of samples used for initialization (we use the default value which should be set to 3 * n_clusters)
        # max_no_improvements denotes the maximum number of no improvement on inertia for the algorithm to terminate early
        mbk = MiniBatchKMeans(init = 'k-means++', n_clusters = n_clusters, batch_size = batch_size, n_init = n_init, max_no_improvement = max_no_improvement, max_iter = max_iter)

        t0 = time.time()

        mbk.fit(self.dataset)
        labels = pairwise_distances_argmin(self.dataset, mbk.cluster_centers_)

        execution_time = time.time() - t0
        
        return labels, mbk.cluster_centers_, mbk.inertia_, execution_time

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

    # Reads an svmlight file
    def load_dataset_from_svmlight(self, path: str, dtype: np.dtype) -> None:

        if not osp.exists(path):
            raise FileNotFoundError

        # We can discard the target labels since clustering is unsupervised
        data, _ = load_svmlight_file(f = path, dtype = dtype, zero_based = True)

        self.dataset = data[:,2:]
        self.vertex_identifier = data[:,0:2]
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
    
    # Write array of given labels into a file for storage (this is done in a human-readable/non-binary way for convenience)
    # NOTE: labels has to be a 1D or 2D array
    def write_clustering_labels(self, labels: np.array, path: str, comment: str) -> None:
        np.savetxt(fname = path, X = labels, comments = '#', fmt = '%d', header = comment)

#Test zone
def cluster_molhiv(cluster_algs: List[Clustering_Algorithm], draw_res: bool, write_res: bool) -> None:
    clusterer = Vertex_Partition_Clustering()

    path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'OGB')
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
        max_eps = np.inf
        n_jobs = None

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