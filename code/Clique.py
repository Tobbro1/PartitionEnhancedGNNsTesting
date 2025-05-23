import os
import sys

import numpy as np
import scipy.sparse.csgraph

from sklearn import metrics
from ast import literal_eval

# Clique algorithm from https://github.com/georgekatona/Clique

class Cluster:
    def __init__(self, dense_units, dimensions, data_point_ids):
        self.id = None
        self.dense_units = dense_units
        self.dimensions = dimensions
        self.data_point_ids = data_point_ids

    def __str__(self):
        return "Dense units: " + str(self.dense_units.tolist()) + "\nDimensions: " \
               + str(self.dimensions) + "\nCluster size: " + str(len(self.data_point_ids)) \
               + "\nData points:\n" + str(self.data_point_ids) + "\n"

# Inserts joined item into candidates list only if its dimensionality fits
def insert_if_join_condition(candidates, item, item2, current_dim):
    joined = item.copy()
    joined.update(item2)
    if (len(joined.keys()) == current_dim) & (not candidates.__contains__(joined)):
        candidates.append(joined)

# Prune all candidates, which have a (k-1) dimensional projection not in (k-1) dim dense units
def prune(candidates, prev_dim_dense_units):
    for c in candidates:
        if not subdims_included(c, prev_dim_dense_units):
            candidates.remove(c)

def subdims_included(candidate, prev_dim_dense_units):
    for feature in candidate:
        projection = candidate.copy()
        projection.pop(feature)
        if not prev_dim_dense_units.__contains__(projection):
            return False
    return True

def self_join(prev_dim_dense_units, dim):
    candidates = []
    for i in range(len(prev_dim_dense_units)):
        for j in range(i + 1, len(prev_dim_dense_units)):
            insert_if_join_condition(
                candidates, prev_dim_dense_units[i], prev_dim_dense_units[j], dim)
    return candidates

def is_data_in_projection(tuple, candidate, xsi):
    if not isinstance(tuple, np.ndarray):
        # We need to access [0] since if tuple is a scarse matrix, it is a 1xnum_f array instead of a num_f array
        tuple = tuple.toarray()[0]
    for feature_index, range_index in candidate.items():
        feature_value = tuple[feature_index]
        # if not isinstance(feature_value, np.ndarray):
        #     feature_value = feature_value.toarray()
        if int(feature_value * xsi % xsi) != range_index:
            return False
    return True

def get_dense_units_for_dim(data, prev_dim_dense_units, dim, xsi, tau):
    candidates = self_join(prev_dim_dense_units, dim)
    prune(candidates, prev_dim_dense_units)

    # Count number of elements in candidates
    projection = np.zeros(len(candidates))
    number_of_data_points = np.shape(data)[0]
    for dataIndex in range(number_of_data_points):
        for i in range(len(candidates)):
            if is_data_in_projection(data[dataIndex], candidates[i], xsi):
                projection[i] += 1
    # print("projection: ", projection)

    # Return elements above density threshold
    is_dense = projection > tau * number_of_data_points
    # print("is_dense: ", is_dense)
    return np.array(candidates)[is_dense]

# dense_units is a list of dicts of the dense units in the form of {f: xsi}
def build_graph_from_dense_units(dense_units):
    graph = np.identity(len(dense_units))
    for i in range(len(dense_units)):
        for j in range(len(dense_units)):
            graph[i, j] = get_edge(dense_units[i], dense_units[j])
    # graph is an adjencency matrix with i,j = 1 iff the i-th and j-th dense unit are neighbors (i.e. the sum of differences (i.e. the distance between the intervals) over all features is at most 1)
    return graph

def get_edge(node1, node2):
    # dim = len(node1)
    distance = 0

    if node1.keys() != node2.keys():
        return 0

    for feature in node1.keys():
        # distance between the cells asociated with feature
        distance += abs(node1[feature] - node2[feature])
        if distance > 1:
            return 0

    return 1


def get_cluster_data_point_ids(data, cluster_dense_units, xsi):
    point_ids = set()

    # Loop through all dense unit
    for u in cluster_dense_units:
        tmp_ids = set(range(np.shape(data)[0]))
        # Loop through all dimensions of dense unit
        for feature_index, range_index in u.items():
            if not isinstance(data, np.ndarray):
                tmp_ids = tmp_ids & set(np.where(np.floor(data[:, feature_index].toarray() * xsi % xsi) == range_index)[0])
            else:
                tmp_ids = tmp_ids & set(np.where(np.floor(data[:, feature_index] * xsi % xsi) == range_index)[0])
        point_ids = point_ids | tmp_ids

    return point_ids

def get_clusters(dense_units, data, xsi):
    graph = build_graph_from_dense_units(dense_units)
    number_of_components, component_list = scipy.sparse.csgraph.connected_components(graph, directed=False)

    dense_units = np.array(dense_units)
    clusters = []
    # For every cluster
    for i in range(number_of_components):
        # Get dense units of the cluster
        cluster_dense_units = dense_units[np.where(component_list == i)]
        # print("cluster_dense_units: ", cluster_dense_units.tolist())

        # Get dimensions of the cluster
        dimensions = set()
        for u in cluster_dense_units:
            dimensions.update(u.keys())

        # Get points of the cluster
        cluster_data_point_ids = get_cluster_data_point_ids(data, cluster_dense_units, xsi)
        # Add cluster to list
        clusters.append(Cluster(cluster_dense_units,
                                dimensions, cluster_data_point_ids))

    return clusters

def get_one_dim_dense_units(data: np.ndarray, tau, xsi):

    number_of_data_points, number_of_features = data.shape
    projection = np.zeros((xsi, number_of_features))
    for f in range(number_of_features):
        for element in data[:, f]:
            # projection[i,f] is the number of datapoints that have the given feature f in the intervall given by i (if a value is shifted by 1/xsi (the intervall size), int(value * xsi % xsi) increases by one)
            if not isinstance(element, np.ndarray):
                element = element.toarray()
            projection[int(element * xsi % xsi), f] += 1
    # print("1D projection:\n", projection, "\n")
    is_dense = projection > tau * number_of_data_points # returns a mask of the pairs interval_idx, feature_idx which is considered dense
    # print("is_dense:\n", is_dense)
    one_dim_dense_units = []
    for f in range(number_of_features):
        for unit in range(xsi):
            if is_dense[unit, f]:
                dense_unit = dict({f: unit})
                one_dim_dense_units.append(dense_unit)
    return one_dim_dense_units

def normalize_features(data):
    normalized_data = data
    number_of_features = np.shape(normalized_data)[1]
    for f in range(number_of_features):
        normalized_data[:, f] -= min(normalized_data[:, f]) - 1e-5
        normalized_data[:, f] *= 1 / (max(normalized_data[:, f]) + 1e-5)
    return normalized_data

# xsi is the number of intervals of equal length each dimension is divided into
# tau is the ratio of datapoints in a cell (of the total amount of data points) required for it to be considered dense
def run_clique(data: np.ndarray, xsi, tau):
    # Finding 1 dimensional dense units
    dense_units = get_one_dim_dense_units(data, tau, xsi)

    # Getting 1 dimensional clusters
    clusters = get_clusters(dense_units, data, xsi)

    # Finding dense units and clusters for dimension > 2
    current_dim = 2
    number_of_features = np.shape(data)[1]
    while (current_dim <= number_of_features) & (len(dense_units) > 0):
        # print("\n", str(current_dim), " dimensional clusters:")
        dense_units = get_dense_units_for_dim(data, dense_units, current_dim, xsi, tau)
        for cluster in get_clusters(dense_units, data, xsi):
            clusters.append(cluster)
        current_dim += 1

    return clusters