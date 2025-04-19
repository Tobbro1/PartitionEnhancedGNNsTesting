# general imports
from typing import List, Optional, Tuple
import time

# pytorch geometric
from torch_geometric.data import Dataset, Data

#numpy
import numpy as np

# import own functionality
from feature_generator import Feature_Generator, TimeLoggingEvent
from util import gen_r_s_ring
import SP_features as spf

class R_S_Ring_SP_Feature_Generator(Feature_Generator):

    #dataset: The dataset of which the features should be generated, k: the k-value for the k-disk generation, 
    #properties_path: if set specifies path of a json file including the dataset properties such that they do not need to be computed again, 
    #write_properties_path: if set specifies the path to which computed properties are saved in a json file; this path is recommended to coincide with the path, the dataset is saved to
    #write_properties_filename: if specified set the filename for the properties json file
    def __init__(self, dataset: Dataset, r: int, s: int, node_pred: bool, samples: Optional[List[int]], absolute_path_prefix: str, dataset_write_path: str, dataset_desc: str, use_editmask: bool, result_mmap_dest: str, editmask_mmap_dest: str, properties_path: Optional[str] = None, write_properties_root_path: Optional[str] = None, write_properties_filename: Optional[str] = None, idx_lookup_path: Optional[str] = None, write_idx_lookup_path: Optional[str] = None, write_idx_lookup_filename: Optional[str] = None):
        super().__init__(dataset = dataset, node_pred = node_pred, samples = samples, absolute_path_prefix = absolute_path_prefix, dataset_write_path = dataset_write_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_dest, editmask_mmap_dest = editmask_mmap_dest, properties_path = properties_path, write_properties_root_path = write_properties_root_path, write_properties_filename = write_properties_filename, idx_lookup_path = idx_lookup_path, write_idx_lookup_path = write_idx_lookup_path, write_idx_lookup_filename = write_idx_lookup_filename)

        # sanity checks
        assert r > 0
        assert s > 0
        assert s > r

        self.r = r
        self.s = s

        # The property distances_alphabet is required for the SP features
        self.prop_manager.properties["distances_alphabet"] = list(range(2*self.s + 2))
        # We add -1 to the possible distances to denote a pair of labels which are disconnected.
        self.prop_manager.properties["distances_alphabet"].append(-1)
        
        self.title_str = f"{self.r}-{self.s}-Ring SP features"

        assert "distances_alphabet" in self.prop_manager.properties
        assert len(self.prop_manager.properties["distances_alphabet"]) > 0

        # remove duplicates if necessary
        self.prop_manager.properties["distances_alphabet"] = list(set(self.prop_manager.properties["distances_alphabet"]))
        self.prop_manager.properties["distances_alphabet"].sort()

        # create a dataset np array as shared memory to store the computed feature vectors
        self.shared_dataset_result_dtype = 'float64'
        self.shared_dataset_result_shape = tuple([len(self.samples), ((len(self.prop_manager.properties["label_alphabet"])**2) * len(self.prop_manager.properties["distances_alphabet"])) + 2])
        self.shared_dataset_result_mmap_dest = result_mmap_dest

        dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'w+', shape = self.shared_dataset_result_shape)
        dataset_result.fill(0)
        dataset_result.flush()

        if self.use_editmask:
            # implement the editmask as a memmap
            self.shared_editmask_mmap_dest = editmask_mmap_dest
            self.shared_editmask_shape = tuple([((len(self.prop_manager.properties["label_alphabet"])**2) * len(self.prop_manager.properties["distances_alphabet"])) + 2,])
            self.shared_editmask_dtype = np.bool

            editmask = np.memmap(self.shared_editmask_mmap_dest, dtype=self.shared_editmask_dtype, mode = 'w+', shape=self.shared_editmask_shape)
            editmask.fill(False)
            editmask[0] = True
            editmask[1] = True
            editmask.flush()
        
        #create graph features object for computation purposes
        self.sp_features = spf.SP_graph_features(label_alphabet = self.prop_manager.properties["label_alphabet"], distances_alphabet = self.prop_manager.properties["distances_alphabet"])

        self.metadata["result_prop"]["feature_desc"] = self.title_str
        self.metadata["result_prop"]["feature_identifier"]["id"] = "r-s-ring_sp"
        self.metadata["result_prop"]["feature_identifier"]["r"] = f"{r}"
        self.metadata["result_prop"]["feature_identifier"]["s"] = f"{s}"

    # Start method of a new process
    # Compute the r-s-ring SP feature vector for a single data point
    def compute_feature(self, idx: int):
        t0 = time.time()

        #get the graph associated with vertex_idx
        vertex_identifier = self.prop_manager.get_vertex_identifier_from_dataset_idx(idx)
        graph_id = vertex_identifier[0]
        vertex_id = vertex_identifier[1]
        cur_graph = self.dataset.get(graph_id)

        #generate r-s-ring
        r_s_ring = gen_r_s_ring(r = self.r, s = self.s, graph_data = cur_graph, vertex = vertex_id)

        #run floyd warshall on the k disk to compute a distances matrix
        floyd_warshall_distances = self.sp_features.floyd_warshall(graph = r_s_ring)

        #compute a dictionary representing the sp distances in the k disk
        sp_map = self.sp_features.sp_feature_map(distances = floyd_warshall_distances, x = r_s_ring.x)
        result, editmask_res = self.sp_features.sp_feature_vector_from_feature_map(dict = sp_map, vertex_identifier = vertex_identifier)

        database_idx = self.prop_manager.get_database_idx_from_vertex_identifier(vertex_identifier = vertex_identifier)

        calc_time = time.time() - t0

        return database_idx, result, editmask_res, calc_time

    # Start method of a new process
    # Compute the r-s-ring SP feature vector for a single data point with logging the times using the log_time() method
    # NOTE: logging the computation times is significantly slower than running the computation without, thus this method should only be utilised on a limited amount of vertices
    def compute_feature_log_times(self, idx: int):
        t0 = time.time()

        self.times = np.full(shape = (self.num_events), dtype = np.float32, fill_value = -1)

        #get the graph associated with vertex_idx
        get_graph_start = time.time()
        vertex_identifier = self.prop_manager.get_vertex_identifier_from_dataset_idx(idx)
        graph_id = vertex_identifier[0]
        vertex_id = vertex_identifier[1]
        cur_graph = self.dataset.get(graph_id)
        get_graph_time = time.time() - get_graph_start
        self.log_time(event = TimeLoggingEvent.load_graph, value = get_graph_time)

        #generate k disk
        gen_r_s_ring_start = time.time()
        r_s_ring = gen_r_s_ring(r = self.r, s = self.s, graph_data = cur_graph, vertex = vertex_id)
        gen_r_s_ring_time = time.time() - gen_r_s_ring_start
        self.log_time(event = TimeLoggingEvent.gen_r_s_ring, value = gen_r_s_ring_time)

        #run floyd warshall on the k disk to compute a distances matrix
        floyd_warshall_start = time.time()
        floyd_warshall_distances = self.sp_features.floyd_warshall(graph = r_s_ring)
        floyd_warshall_time = time.time() - floyd_warshall_start
        self.log_time(event= TimeLoggingEvent.floyd_warshall, value = floyd_warshall_time)

        #compute a dictionary representing the sp distances in the k disk
        sp_map_start = time.time()
        sp_map = self.sp_features.sp_feature_map(distances = floyd_warshall_distances, x = r_s_ring.x)
        result, editmask_res = self.sp_features.sp_feature_vector_from_feature_map(dict = sp_map, vertex_identifier = tuple([graph_id, vertex_id]))
        sp_map_time = time.time() - sp_map_start
        self.log_time(event = TimeLoggingEvent.SP_vector_computation, value = sp_map_time)

        save_res_start = time.time()
        database_idx = self.prop_manager.get_database_idx_from_vertex_identifier(vertex_identifier = vertex_identifier)
        save_res_time = time.time() - save_res_start
        self.log_time(event= TimeLoggingEvent.get_database_idx, value = save_res_time)

        calc_time = time.time() - t0

        return database_idx, result, editmask_res, calc_time, self.times
    

# Function to compute a single r-s-ring sp features vector
# incompatible with editmasks
# Returns: array of the feature vector, calculation time
def compute_single_r_s_ring_sp_feature_vector(graph: Data, vertex_id: int, r: int, s: int, sp_features: spf.SP_graph_features) -> Tuple[np.array, float]:
    # untested

    t0 = time.time()

    vertex_identifier = tuple([0,0])

    #generate k disk
    r_s_ring = gen_r_s_ring(r = r, s = s, graph_data = graph, vertex = vertex_id)

    #run floyd warshall on the k disk to compute a distances matrix
    floyd_warshall_distances = sp_features.floyd_warshall(graph = r_s_ring)

    #compute a dictionary representing the sp distances in the k disk
    sp_map = sp_features.sp_feature_map(distances = floyd_warshall_distances, x = r_s_ring.x)
    result, _ = sp_features.sp_feature_vector_from_feature_map(dict = sp_map, vertex_identifier = vertex_identifier)

    return result[2:], time.time() - t0
