# General imports
from typing import List, Optional, Tuple
import time

# numpy
import numpy as np

# pytorch geometric
from torch_geometric.data import Dataset, Data

#import own functionality
from feature_generator import Feature_Generator, TimeLoggingEvent
import SP_features as spf

# NOTE: This feature should only be computed for sufficiently small graphs since it computes the distance matrix of the whole graph with the floyd warshall algorithm (O(n^3))
class Vertex_SP_Feature_Generator(Feature_Generator):
    def __init__(self, dataset: Dataset, node_pred: bool, samples: Optional[List[int]], absolute_path_prefix: str, dataset_write_path: str, dataset_desc: str, use_editmask: bool, result_mmap_dest: str, editmask_mmap_dest: str, properties_path: Optional[str] = None, write_properties_root_path: Optional[str] = None, write_properties_filename: Optional[str] = None, idx_lookup_path: Optional[str] = None, write_idx_lookup_path: Optional[str] = None, write_idx_lookup_filename: Optional[str] = None):
        super().__init__(dataset = dataset, node_pred = node_pred, samples = samples, absolute_path_prefix = absolute_path_prefix, dataset_write_path = dataset_write_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_dest, editmask_mmap_dest = editmask_mmap_dest, properties_path = properties_path, write_properties_root_path = write_properties_root_path, write_properties_filename = write_properties_filename, idx_lookup_path = idx_lookup_path, write_idx_lookup_path = write_idx_lookup_path, write_idx_lookup_filename = write_idx_lookup_filename)

        # No two vertices have distance 0 as we disallow self-loops
        self.prop_manager.properties["distances_alphabet"] = list(range(1, max(self.prop_manager.properties["graph_sizes"]) - 1))
        # For unreachable vertex pairs
        self.prop_manager.properties["distances_alphabet"].append(-1)

        self.title_str = f"Vertex SP features"

        assert "distances_alphabet" in self.prop_manager.properties
        assert len(self.prop_manager.properties["distances_alphabet"]) > 0

        # remove duplicates if necessary
        self.prop_manager.properties["distances_alphabet"] = list(set(self.prop_manager.properties["distances_alphabet"]))
        self.prop_manager.properties["distances_alphabet"].sort()

        # create a dataset np array as a memmap to store the computed feature vectors
        self.shared_dataset_result_dtype = 'float64'
        self.shared_dataset_result_shape = tuple([len(self.samples), (len(self.prop_manager.properties["label_alphabet"]) * len(self.prop_manager.properties["distances_alphabet"])) + 2])
        self.shared_dataset_result_mmap_dest = result_mmap_dest

        dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'w+', shape = self.shared_dataset_result_shape)
        dataset_result.fill(0)
        dataset_result.flush()

        if self.use_editmask:
            # implement the editmask as a memmap
            self.shared_editmask_mmap_dest = editmask_mmap_dest
            self.shared_editmask_shape = tuple([(len(self.prop_manager.properties["label_alphabet"]) * len(self.prop_manager.properties["distances_alphabet"])) + 2,])
            self.shared_editmask_dtype = np.bool

            editmask = np.memmap(self.shared_editmask_mmap_dest, dtype=self.shared_editmask_dtype, mode = 'w+', shape=self.shared_editmask_shape)
            editmask.fill(False)
            editmask[0] = True
            editmask[1] = True
            editmask.flush()

        #create vertex features object for computation purposes
        self.sp_features = spf.SP_vertex_features(label_alphabet = self.prop_manager.properties["label_alphabet"], distances_alphabet = self.prop_manager.properties["distances_alphabet"])

        self.metadata["result_prop"]["feature_desc"] = self.title_str
        self.metadata["result_prop"]["feature_identifier"]["id"] = "vertex_sp"

    # Start method of a new process
    # Compute the Vertex SP feature vectors for all vertices of a graph specified by idx
    def compute_feature(self, idx: int):
        t0 = time.time()

        graph_id = idx
        cur_graph = self.dataset.get(graph_id)
        num_vertices = cur_graph.num_nodes

        # initialize an array of indices of the vertices included in the given graph
        database_idx = np.zeros(shape = (num_vertices,), dtype = int)
        # initialize an array of result vectors for the vertices included in the given graph
        result = np.zeros(shape = (num_vertices, self.shared_dataset_result_shape[1]), dtype = self.shared_dataset_result_dtype)
        # initialize an array representing the combined editmask vector for all vertices of the given graph
        editmask_res = np.full(shape = self.shared_editmask_shape, dtype = self.shared_editmask_dtype, fill_value = False)

        # run floyd warshall to compute a distances matrix for the graph
        floyd_warshall_distances = self.sp_features.floyd_warshall(graph = cur_graph)

        # for each vertex of the given graph we compute the feature vector
        for vertex_id in range(num_vertices):
            vertex_identifier = tuple([graph_id, vertex_id])

            vertex_sp_map = self.sp_features.vertex_sp_feature_map(distances = floyd_warshall_distances, x = cur_graph.x, vertex_id = vertex_id)
            result_v, editmask_v = self.sp_features.vertex_sp_feature_vector_from_map(dict = vertex_sp_map, vertex_identifier = vertex_identifier)

            editmask_res = np.logical_or(editmask_res, editmask_v)

            database_idx[vertex_id] = self.prop_manager.get_database_idx_from_vertex_identifier(vertex_identifier = vertex_identifier)
            result[vertex_id,:] = result_v

        calc_time = time.time() - t0

        return database_idx, result, editmask_res, calc_time
    
    # Helper to evaluate time complexity of individual operations
    # Override
    def log_time(self, event: TimeLoggingEvent, value: float, vertex_identifier: Tuple[int, int]):
        
        self.times[vertex_identifier[1], event.value] = value

    # Start method of a new process
    # Compute the Vertex SP feature vectors for all vertices of a graph specified by idx
    def compute_feature_log_times(self, idx: int):

        t0 = time.time()

        get_graph_start = time.time()
        graph_id = idx
        cur_graph = self.dataset.get(graph_id)
        num_vertices = cur_graph.num_nodes

        self.times = np.full(shape = (num_vertices, self.num_events), dtype = np.float32, fill_value = -1)

        # initialize an array of indices of the vertices included in the given graph
        database_idx = np.zeros(shape = (num_vertices,), dtype = int)
        # initialize an array of result vectors for the vertices included in the given graph
        result = np.zeros(shape = (num_vertices, self.shared_dataset_result_shape[1]), dtype = self.shared_dataset_result_dtype)
        # initialize an array representing the combined editmask vector for all vertices of the given graph
        editmask_res = np.full(shape = self.shared_editmask_shape, dtype = self.shared_editmask_dtype, fill_value = False)

        get_graph_time = time.time() - get_graph_start
        for v in range(num_vertices):
            self.log_time(event = TimeLoggingEvent.load_graph, value = get_graph_time, vertex_identifier = tuple([graph_id, v]))

        # run floyd warshall to compute a distances matrix for the graph
        floyd_warshall_start = time.time()
        floyd_warshall_distances = self.sp_features.floyd_warshall(graph = cur_graph)
        floyd_warshall_time = time.time() - floyd_warshall_start
        for v in range(num_vertices):
            self.log_time(event = TimeLoggingEvent.floyd_warshall, value = floyd_warshall_time, vertex_identifier = tuple([graph_id, v]))

        # for each vertex of the given graph we compute the feature vector
        for vertex_id in range(num_vertices):
            sp_map_start = time.time()

            vertex_identifier = tuple([graph_id, vertex_id])

            vertex_sp_map = self.sp_features.vertex_sp_feature_map(distances = floyd_warshall_distances, x = cur_graph.x, vertex_id = vertex_id)
            result_v, editmask_v = self.sp_features.vertex_sp_feature_vector_from_map(dict = vertex_sp_map, vertex_identifier = vertex_identifier)

            editmask_res = np.logical_or(editmask_res, editmask_v)

            database_idx[vertex_id] = self.prop_manager.get_database_idx_from_vertex_identifier(vertex_identifier = vertex_identifier)
            result[vertex_id,:] = result_v

            sp_map_time = time.time() - sp_map_start
            self.log_time(event = TimeLoggingEvent.SP_vector_computation, value = sp_map_time, vertex_identifier = vertex_identifier)

        calc_time = time.time() - t0

        return database_idx, result, editmask_res, calc_time, self.times
    
# Function to compute all vertex sp features vectors of a given graph
# incompatible with editmasks
# Returns: list of arrays representing the feature vectors, calculation time
def compute_vertex_sp_feature_vectors(graph: Data, sp_features: spf.SP_vertex_features) -> Tuple[List[np.array], float]:
    # untested

    t0 = time.time()

    graph_id = 0
    num_vertices = graph.num_nodes

    result = []

    # run floyd warshall to compute a distances matrix for the graph
    floyd_warshall_distances = sp_features.floyd_warshall(graph = graph)

    # for each vertex of the given graph we compute the feature vector
    for vertex_id in range(num_vertices):
        vertex_identifier = tuple([graph_id, vertex_id])

        vertex_sp_map = sp_features.vertex_sp_feature_map(distances = floyd_warshall_distances, x = graph.x, vertex_id = vertex_id)
        result_v, _ = sp_features.vertex_sp_feature_vector_from_map(dict = vertex_sp_map, vertex_identifier = vertex_identifier)

        result.append(result_v[2:])

    return result, time.time() - t0