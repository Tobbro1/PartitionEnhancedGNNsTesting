# general imports
from typing import List, Optional, Tuple
import time

# pytorch geometric
from torch_geometric.data import Dataset, Data

# numpy
import numpy as np

# import own functionality
from feature_generator import Feature_Generator, TimeLoggingEvent
from Lovacs_features import Lovasz_graph_features
from util import gen_k_disk

class K_Disk_LO_Feature_Generator(Feature_Generator):

    #dataset: The dataset of which the features should be generated, k: the k-value for the k-disk generation, 
    #properties_path: if set specifies path of a json file including the dataset properties such that they do not need to be computed again, 
    #write_properties_path: if set specifies the path to which computed properties are saved in a json file; this path is recommended to coincide with the path, the dataset is saved to
    #write_properties_filename: if specified set the filename for the properties json file
    def __init__(self, dataset: Dataset, k: int, lo_graph_sizes_range: Tuple[int,int], lo_s: int, node_pred: bool, samples: Optional[List[int]], absolute_path_prefix: str, dataset_write_path: str, dataset_desc: str, use_editmask: bool, result_mmap_dest: str, editmask_mmap_dest: str, properties_path: Optional[str] = None, write_properties_root_path: Optional[str] = None, write_properties_filename: Optional[str] = None, idx_lookup_path: Optional[str] = None, write_idx_lookup_path: Optional[str] = None, write_idx_lookup_filename: Optional[str] = None):
        super().__init__(dataset = dataset, node_pred = node_pred, samples = samples, absolute_path_prefix = absolute_path_prefix, dataset_write_path = dataset_write_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_dest, editmask_mmap_dest = editmask_mmap_dest, properties_path = properties_path, write_properties_root_path = write_properties_root_path, write_properties_filename = write_properties_filename, idx_lookup_path = idx_lookup_path, write_idx_lookup_path = write_idx_lookup_path, write_idx_lookup_filename = write_idx_lookup_filename)

        #sanity checks
        assert k > 0
        assert lo_s > 0
        assert lo_graph_sizes_range[0] > 0 and lo_graph_sizes_range[1] >= lo_graph_sizes_range[0]

        self.k = k
        self.lo_graph_sizes_range = lo_graph_sizes_range
        self. num_features = (self.lo_graph_sizes_range[1] - self.lo_graph_sizes_range[0]) + 1
        self.lo_s = lo_s

        self.embedding_dim = int(self.prop_manager.properties["graph_size"]["max"]) + 1

        self.title_str = f"{self.k}-Disk Lovasz features"

        # create a dataset np array as shared memory to store the computed feature vectors
        self.shared_dataset_result_dtype = 'float64'
        self.shared_dataset_result_shape = tuple([len(self.samples), self.num_features])
        self.shared_dataset_result_mmap_dest = result_mmap_dest

        dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'w+', shape = self.shared_dataset_result_shape)
        dataset_result.fill(0)
        dataset_result.flush()

        if self.use_editmask:
            # implement the editmask as a memmap
            self.shared_editmask_mmap_dest = editmask_mmap_dest
            self.shared_editmask_shape = tuple([self.num_features,])
            self.shared_editmask_dtype = np.bool

            editmask = np.memmap(self.shared_editmask_mmap_dest, dtype=self.shared_editmask_dtype, mode = 'w+', shape=self.shared_editmask_shape)
            editmask.fill(False)
            editmask[0] = True
            editmask[1] = True
            editmask.flush()
        
        #create graph features object for computation purposes
        self.lo_features = Lovasz_graph_features(embedding_dim = self.embedding_dim)

        self.metadata["result_prop"]["feature_desc"] = self.title_str
        self.metadata["result_prop"]["feature_identifier"]["id"] = "k-disk_lo"
        self.metadata["result_prop"]["feature_identifier"]["k"] = f"{k}"
        self.metadata["result_prop"]["feature_identifier"]["lo_graph_sizes_range"] = list(lo_graph_sizes_range)
        self.metadata["result_prop"]["feature_identifier"]["lo_s"] = lo_s

    # Start method of a new process
    # Compute the k-disk Lovasz feature vector for a single data point.
    # vertex_idx is a dataset_idx, NOT a database_idx
    def compute_feature(self, idx: int):
        t0 = time.time()

        #get the graph associated with vertex_idx
        vertex_identifier = self.prop_manager.get_vertex_identifier_from_dataset_idx(idx)
        graph_id = vertex_identifier[0]
        vertex_id = vertex_identifier[1]
        cur_graph = self.dataset.get(graph_id)

        #generate k disk
        k_disk = gen_k_disk(k = self.k, graph_data = cur_graph, vertex = vertex_id)

        # generate an orthonormal representation of the k-disk
        theta, S = self.lo_features.compute_lovasz_number(graph = k_disk, num_vertices = k_disk.num_nodes)
        orthonormal_rep = self.lo_features.compute_orthonormal_basis(theta = theta, S = S, embedding_dim = self.embedding_dim)

        # compute the lo feature vector
        result = self.lo_features.compute_lovasz_feature_vector(vertex_identifier = vertex_identifier, graph_size_range = self.lo_graph_sizes_range, orthonormal_representation = orthonormal_rep, num_samples = self.lo_s)
        
        # editmask is not implemented for lo features
        editmask = np.full(shape = (self.num_features,), dtype = bool, fill_value = True)

        database_idx = self.prop_manager.get_database_idx_from_vertex_identifier(vertex_identifier = vertex_identifier)

        calc_time = time.time() - t0

        return database_idx, result, editmask, calc_time

    def compute_feature_log_times(self, idx: int):
        raise NotImplementedError
