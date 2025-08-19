from typing import Optional, Tuple, Dict, List

import os.path as osp
import numpy as np
import os

import matplotlib.pyplot as plt
import colorsys
from matplotlib.markers import MarkerStyle

import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset

from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

from SP_features import SP_vertex_features

from developmentHelpers import drawGraph
from util import write_numpy_txt

# Helper for clustering
# Returns a tuple of assignments, centroids, inertia and execution time of the clustering
def mini_batch_k_means(data: np.ndarray, n_clusters: int) -> np.ndarray:

    n_clusters = n_clusters
    batch_size = 128
    n_init = 1
    max_no_improvement = 20
    max_iter = 20

    # batch-size greater than 256 * num_cores for paralllelism on all cores
    # n_init describes the number of random initializations tried, before using the best (in terms of inertia) initialization (note that the algorithm is not run for every initialization but only for the best)
    # init_size is the number of samples used for initialization (we use the default value which should be set to 3 * n_clusters)
    # max_no_improvements denotes the maximum number of no improvement on inertia for the algorithm to terminate early
    init_method = 'k-means++'
    init_size = 3 * batch_size
    if init_size < n_clusters:
        init_size = 3 * n_clusters
    reassignment_ratio = 0.01
    tol = 0.0
    
    mbk = MiniBatchKMeans(n_clusters = n_clusters, init = init_method, max_iter = max_iter, batch_size = batch_size, tol = tol, max_no_improvement = max_no_improvement, init_size = init_size, n_init = n_init, reassignment_ratio = reassignment_ratio)
    mbk.fit(data)
    labels = pairwise_distances_argmin(data, mbk.cluster_centers_)

    centroids = mbk.cluster_centers_
    
    inertia = mbk.inertia_
    
    return labels, centroids, inertia

# Generate some plots and vectors for a graphic in the presentation
def gen_training_pipeline(graphs: List[Data], draw_colors: List[bool], label_alphabet: List[int], distances_alphabet: List[int]) -> None:
    
    default_path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir, 'other')
    markers = ['v', '^']
    color_graphs = [graphs[i] for i in range(len(graphs)) if draw_colors[i]]

    # Generate colors for the graphs that should be colored when clustering
    colors = []
    num_colors = 0
    for graph in color_graphs:
        num_colors += graph.num_nodes

    colors = [colorsys.hsv_to_rgb(x*1.0/num_colors, 1, 0.8) for x in range(num_colors)]
    # Add gray for all graphs that should not be colored
    colors.append(colorsys.hsv_to_rgb(h = 0, s = 0, v = 0.5))

    # Save the RGB codes for the colors by converting them to an array before storing it
    color_res = np.empty(shape = (len(colors), 3), dtype = np.float64)
    for color_idx in range(len(colors)):
        color_res[color_idx,:] = np.array(list(colors[color_idx]), dtype = np.float64)
    write_numpy_txt(path = default_path, filename = 'training_pipeline_colors.txt', data = color_res, comment = None)

    # Generate vertex SP feature vectors for the graphs
    vsp_gen = SP_vertex_features(label_alphabet = label_alphabet, distances_alphabet = distances_alphabet)

    result_array = np.empty((1, (len(label_alphabet) * len(distances_alphabet)) + 2), dtype = np.int64)

    for idx in range(len(graphs)):
        cur_graph = graphs[idx]
        distances = vsp_gen.floyd_warshall(cur_graph)
        for vertex_idx in range(cur_graph.num_nodes):
            vsp_map = vsp_gen.vertex_sp_feature_map(distances = distances, x = cur_graph.x, vertex_id = vertex_idx)
            result, _ = vsp_gen.vertex_sp_feature_vector_from_map(dict = vsp_map, vertex_identifier = (idx, vertex_idx))
            if idx == 0 and vertex_idx == 0:
                # First entry
                result_array[0,:] = result.astype(np.int64)
            else:
                # Concatenate
                result_array = np.concatenate((result_array, result.reshape((1, result.shape[0])).astype(np.int64)), axis = 0)

    # Save the computed feature vectors
    write_numpy_txt(path = default_path, filename = 'training_pipeline_vectors.txt', data = result_array, comment = None, format = '%1.1d')

    # Draw the input graphs
    for idx in range(len(graphs)):
        labels = torch.tensor(np.arange(0, graphs[idx].num_nodes))
        drawGraph(graph = graphs[idx], figure_count = idx + 1, labels = labels)

    # PCA to reduce dimensionality to 2 for drawing
    draw_pca = TruncatedSVD(n_components = 2, algorithm = 'arpack')
    draw_pca.fit(result_array[:,2:])

    reduced_data = draw_pca.transform(result_array[:,2:])

    # Cluster the reduced feature vectors
    n_clusters = 2
    cluster_labels, centroids, inertia = mini_batch_k_means(reduced_data, n_clusters = n_clusters)
    
    # Plot the feature vectors with clusters
    plt.figure(len(graphs) + 1)
    # Generate the vertex IDs of vertices with identical vectors
    unique_dict = {}
    unique_vectors, unique_indices, unique_inverse = np.unique(reduced_data, axis = 0, return_index = True, return_inverse = True)
    for u_idx in range(unique_vectors.shape[0]):
        unique_dict[u_idx] = np.where(unique_inverse == u_idx)[0].tolist()

    cur_start_idx = 0
    for graph_idx in range(len(graphs)):
        cur_graph = graphs[graph_idx]
        for vertex_idx in range(cur_graph.num_nodes):
            idx = cur_start_idx + vertex_idx
            c = cluster_labels[idx]
            if draw_colors[graph_idx]:
                unique_idx = unique_inverse[idx]
                if len(unique_dict[unique_idx]) == 2:
                    if unique_dict[unique_idx][0] == idx:
                        plt.scatter(reduced_data[idx,0], reduced_data[idx,1], s = 200, marker = MarkerStyle(markers[c], fillstyle = "left"), color = colors[idx], alpha = 1)
                    else:
                        plt.scatter(reduced_data[idx,0], reduced_data[idx,1], s = 200, marker = MarkerStyle(markers[c], fillstyle = "right"), color = colors[idx], alpha = 1)
                elif len(unique_dict[unique_idx]) > 2:
                    print(f'Unique Index {unique_idx} has more than 2 representatives: {unique_dict[unique_idx]}')
                    plt.scatter(reduced_data[idx,0], reduced_data[idx,1], s = 200, marker = markers[c], color = colors[idx], alpha = 1)
                else:
                    plt.scatter(reduced_data[idx,0], reduced_data[idx,1], s = 200, marker = markers[c], color = colors[idx], alpha = 1)
            else:
                plt.scatter(reduced_data[idx,0], reduced_data[idx,1], s = 200, marker = markers[c], color = colors[-1], alpha = 1)

        cur_start_idx += cur_graph.num_nodes


if __name__ == '__main__':
    
    # Labels and distances
    label_alphabet = [0]
    distances_alphabet = np.arange(0, 5).tolist()

    # Graph 1:
    num_vertices = 5 # 7
    x1 = torch.tensor([0] * num_vertices)
    # row1 = [0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 4, 5, 5, 6]
    # col1 = [1, 2, 3, 0, 0, 3, 0, 2, 4, 5, 3, 3, 6, 5]
    row1 = [0, 0, 2, 2, 1, 2, 3, 4]
    col1 = [1, 2, 3, 4, 0, 0, 2, 2]
    edge_index1 = torch.tensor([row1, col1])
    data1 = Data(x = x1, edge_index = edge_index1)
    
    print(x1)
    print(x1.size())
    print(edge_index1)
    print(edge_index1.size())

    # Graph 2:
    num_vertices = 4
    x2 = torch.tensor([0] * num_vertices)
    row2 = [0, 0, 0, 1, 1, 2, 3, 2]
    col2 = [1, 2, 3, 2, 0, 0, 0, 1]
    edge_index2 = torch.tensor([row2, col2])
    data2 = Data(x = x2, edge_index = edge_index2)

    graphs = [data1, data2]
    draw_colors = [True, False]

    gen_training_pipeline(graphs = graphs, draw_colors = draw_colors, label_alphabet = label_alphabet, distances_alphabet = distances_alphabet)

    plt.show()