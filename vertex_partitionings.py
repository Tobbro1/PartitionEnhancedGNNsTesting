#TODO:
#Create a class that incorporates functionality to:
#Read the saved feature vectors from a file
#Cluster these feature vectors using different clustering techniques (k-means, OPTICS)
#Include functionality to write the clustering results into a file (a dictionary (graph_id, vertex_id) -> cluster_id ?)

#system
import time
import os.path as osp

#general
from typing import Dict
from typing import Tuple

#scikit-learn
import sklearn
from sklearn.datasets import load_svmlight_file
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.decomposition import PCA

#numpy
import numpy as np 

#plotting
import matplotlib.pyplot as plt

SEED = 37

class Vertex_Partition_Clustering():

    def __init__(self):
        super().__init__()

        self.dataset = None
        self.vertex_identifier = None

        self.num_vertices = -1
        self.num_features = -1

        self.pca = None
        self.draw_pca = None

    # Returns a tuple of the learned components and the ratio of their explained variance
    def generate_pca(self, target_dimensions) -> Tuple[np.ndarray, np.ndarray]:

        # n_components is the number of principal components that are utilised, thus the number of dimensions after the PCA
        # if copy is set to False, the data passed to train the PCA gets overwritten
        # whiten can sometimes improve downstream accuracy of estimators but removes some information from the transformed signal as it normalizes component wise variances
        # svd solver selects the method to solve the underlying singular value decomposition (see PCA documentation for values)
        self.pca = PCA(n_components = target_dimensions, copy = True, svd_solver = 'arpack')
        self.pca.fit(self.dataset)

        return self.pca.components_, self.pca.explained_variance_ratio_

    def apply_pca_to_dataset(self):
        if self.dataset is None or self.num_vertices < 1:
            raise ValueError("Dataset invalid")
        
        if self.pca is None:
            raise ValueError("Initialize PCA before applying")

        self.dataset = self.pca.transform(self.dataset)

    # A PCA to two dimensions used for visualization
    # Returns a tuple of the learned components and the ratio of their explained variance
    def generate_draw_pca(self) -> Tuple[np.ndarray, np.ndarray]:

        self.draw_pca = PCA(n_components = 2, copy = True, svd_solver = 'arpack')
        self.draw_pca.fit(self.dataset)

        return self.draw_pca.components_, self.draw_pca.explained_variance_ratio_

    # This method draws the dataset colored by their assigned cluster according to the label parameter. This makes use of a PCA to two dimensions
    # The grid_granularity parameter is used when visualizing centroid based methods. Note that the grid is made between the smallest and highest values for the PCA transformed vectors which might lead to huge memroy requirements in case of large ranges and small granularity
    def draw_clustering_data(self, num_figure: int, title: str, centroids = None, grid_granularity = 2):

        if self.draw_pca is None:
            self.generate_draw_pca()
        
        reduced_data = self.draw_pca.transform(self.dataset)

        # k-means, code taken from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
        if centroids is not None:
            centroids = self.draw_pca.transform(centroids)

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
            plt.xlabel(f"Component 1: {self.draw_pca.explained_variance_ratio_[0]}")
            plt.ylabel(f"Component 2: {self.draw_pca.explained_variance_ratio_[1]}")
            plt.xticks(())
            plt.yticks(())


    # Returns a tuple of assignments, centroids, inertia and execution time of the clustering
    def mini_batch_k_means(self, n_clusters: int, batch_size: int, n_init: int, max_no_improvement: int, max_iter: int) -> float:

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

    # Reads an svm-light file
    def load_dataset_from_svmlight(self, path: str, dtype: np.dtype):

        if not osp.exists(path):
            raise FileNotFoundError

        # We can discard the target labels
        data, _ = load_svmlight_file(f = path, dtype = dtype, zero_based = True)

        self.dataset = data[:,2:]
        self.vertex_identifier = data[:,0:2]
        self.num_vertices, self.num_features = self.dataset.shape

    # Returns a dictionary mapping (graph_id, vertex_id) to its corresponding cluster_id
    def generate_clustering_result_dict(self, labels) -> Dict[Tuple[int, int], int]:

        result = {}

        for i in range(self.num_vertices):
            _t = tuple([int(self.vertex_identifier[i,0]), int(self.vertex_identifier[i,1])])
            result[_t] = labels[i]

        return result

#Test zone
if __name__ == '__main__':

    #reproducability
    np.random.seed(SEED)

    clusterer = Vertex_Partition_Clustering()

    path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'TU')
    mutag_path = osp.join(path, "MUTAG")
    dd_path = osp.join(path, "DD")
    mutag_dataset_filename = "k_disk_SP_features_MUTAG.svmlight"
    dd_dataset_filename = "k_disk_SP_features_DD.svmlight"
    dataset_path = osp.join(path, mutag_path, mutag_dataset_filename)

    clusterer.load_dataset_from_svmlight(dataset_path, dtype='float64')

    mbk_n_clusters = 10
    mbk_batch_size = 1024
    mbk_n_init = 10
    mbk_max_no_improvement = 10
    mbk_max_iter = 1000

    labels, centroids, inertia, clustering_time = clusterer.mini_batch_k_means(n_clusters = mbk_n_clusters, batch_size = mbk_batch_size, n_init = mbk_n_init, max_no_improvement = mbk_max_no_improvement, max_iter = mbk_max_iter)
    #Computed labels: {labels}\nComputed centroids: {centroids}\n
    
    res = clusterer.generate_clustering_result_dict(labels)
    print(f"Inertia: {inertia}, clustering time: {clustering_time}")

    #draw the clustering results using a 2-dimensional PCA
    clusterer.draw_clustering_data(0, 'Testdrawing', centroids, grid_granularity=0.2)

    plt.show()