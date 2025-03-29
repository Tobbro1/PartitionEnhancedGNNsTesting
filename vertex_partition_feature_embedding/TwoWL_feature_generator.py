#general imports
from typing import List, Optional
import time

from torch_geometric.data import Dataset

#numpy
import numpy as np

#import own functionality
from feature_generator import Feature_Generator, TimeLoggingEvent
import TwoWL_features as twoWL
from util import gen_k_disk, gen_r_s_ring

# This implementation incorporates k-disks as well as r-s-rings
# NOTE: These features require sufficient memory to be stored during the complete calculation.
class TwoWL_Feature_Generator(Feature_Generator):

    # NOTE: If k is not None, this will create a parser for k-disks, if k is None, r and s have to be set to positive integers
    # dataset: The dataset of which the features should be generated, k: the k-value for the k-disk generation, 
    # properties_path: if set specifies path of a json file including the dataset properties such that they do not need to be computed again, 
    # write_properties_path: if set specifies the path to which computed properties are saved in a json file; this path is recommended to coincide with the path, the dataset is saved to
    # write_properties_filename: if specified set the filename for the properties json file
    def __init__(self, dataset: Dataset, k: Optional[int], r: Optional[int], s: Optional[int], num_layers: int, node_pred: bool, samples: Optional[List[int]], dataset_write_path: str, dataset_write_filename: str, result_mmap_dest: str, editmask_mmap_dest: str, properties_path: Optional[str] = None, write_properties_root_path: Optional[str] = None, write_properties_filename: Optional[str] = None):
        super().__init__(dataset = dataset, node_pred = node_pred, samples = samples, dataset_write_path = dataset_write_path, dataset_write_filename = dataset_write_filename, result_mmap_dest = result_mmap_dest, editmask_mmap_dest = editmask_mmap_dest, properties_path = properties_path, write_properties_root_path = write_properties_root_path, write_properties_filename = write_properties_filename)

        # Evaluate whether the k-disks or r-s-rings should be used to compute 2WL features
        if k is not None:
            assert k > 0
            self.k_disk_mode = True
            self.k = k
        elif r is not None and s is not None:
            assert r > 0 and s >= r
            self.k_disk_mode = False
            self.r = r
            self.s = s
        else:
            raise ValueError(f"Either k or r and s have to be initialized.")
        
        assert num_layers > 0
        self.num_layers = num_layers
      
        if self.k_disk_mode:
            self.title_str = f"{self.k}-Disk 2-WL features with {self.num_layers} layers"
        else:
            self.title_str = f"{self.r}-{self.s}-Ring 2-WL features with {self.num_layers} layers"

        # The resulting database cannot be initialized and has to be handled by the parent process after finished computation

        # The editmask implementation is not utilised in 2WL features
        
        #create graph features object for computation purposes
        self.two_wl = twoWL.TwoWL()

    # Start method of a new process
    # Compute the 2WL feature vector of a k-disk or r-s-ring depending on the initialization
    # vertex_idx is a dataset_idx, NOT a database_idx
    def compute_feature(self, idx: int):
        # Get the graph associated with vertex_idx
        vertex_identifier = self.get_vertex_identifier_from_dataset_idx(idx)
        graph_id = vertex_identifier[0]
        vertex_id = vertex_identifier[1]
        cur_graph = self.dataset.get(graph_id)

        # Generate k-disk or r-s-ring
        if self.k_disk_mode:
            vertex_graph = gen_k_disk(k = self.k, graph_data = cur_graph, vertex = vertex_id)
        else:
            vertex_graph = gen_r_s_ring(r = self.r, s = self.s, graph_data = cur_graph, vertex = vertex_id)

        # Compute the coloring of the vertex_graph
        coloring = self.two_wl.compute_two_wl(graph = vertex_graph)

        # Convert the coloring to a dictionary of color frequencies to reduce memroy usage
        color_freq = self.two_wl.compute_color_frequencies(coloring = coloring)

        # Calculate the database idx of the just computed frequencies
        database_idx = self.get_database_idx_from_vertex_identifier(vertex_identifier = vertex_identifier)

        # Return the dictionary, this has to be post-processed
        return database_idx, vertex_identifier, color_freq
    
    # Start method of a new process
    # Compute the 2WL feature vector of a k-disk or r-s-ring depending on the initialization
    # vertex_idx is a dataset_idx, NOT a database_idx
    # NOTE: logging the computation times is significantly slower than running the computation without, thus this method should only be utilised on a limited amount of vertices
    def compute_feature_log_times(self, idx: int):
        self.times = np.full(shape = (self.num_events), dtype = np.float32, fill_value = -1)

        # Get the graph associated with vertex_idx
        get_graph_start = time.time()
        vertex_identifier = self.get_vertex_identifier_from_dataset_idx(idx)
        graph_id = vertex_identifier[0]
        vertex_id = vertex_identifier[1]
        cur_graph = self.dataset.get(graph_id)
        get_graph_time = time.time() - get_graph_start
        self.log_time(event = TimeLoggingEvent.load_graph, value = get_graph_time)


        # Generate k-disk or r-s-ring
        if self.k_disk_mode:
            gen_k_disk_start = time.time()
            vertex_graph = gen_k_disk(k = self.k, graph_data = cur_graph, vertex = vertex_id)
            gen_k_disk_time = time.time() - gen_k_disk_start
            self.log_time(event = TimeLoggingEvent.gen_k_disk, value = gen_k_disk_time)
        else:
            gen_r_s_ring_start = time.time()
            vertex_graph = gen_r_s_ring(r = self.r, s = self.s, graph_data = cur_graph, vertex = vertex_id)
            gen_r_s_ring_time = time.time() - gen_r_s_ring_start
            self.log_time(event = TimeLoggingEvent.gen_r_s_ring, value = gen_r_s_ring_time)

        # Compute the coloring of the vertex_graph
        two_wl_time_start = time.time()
        coloring = self.two_wl.compute_two_wl(graph = vertex_graph)

        # Convert the coloring to a dictionary of color frequencies to reduce memroy usage
        color_freq = self.two_wl.compute_color_frequencies(coloring = coloring)
        two_wl_time = time.time() - two_wl_time_start
        self.log_time(event = TimeLoggingEvent.Two_WL_computation, value = two_wl_time)

        # Calculate the database idx of the just computed frequencies
        save_res_start = time.time()
        database_idx = self.get_database_idx_from_vertex_identifier(vertex_identifier = vertex_identifier)
        save_res_time = time.time() - save_res_start
        self.log_time(event = TimeLoggingEvent.get_database_idx, value = save_res_time)

        # Return the dictionary, this has to be post-processed
        return database_idx, vertex_identifier, color_freq, self.times