#general imports
import multiprocessing.shared_memory
from typing import List, Optional, Tuple
import os
import os.path as osp
import time
import json #to implement saving and loading the properties of datasets without having to run the computation multiple times
import tqdm
from enum import Enum
import atexit

#parallelization
import multiprocessing

#pytorch, pytorch geometric
import torch
from torch import Tensor

from torch_geometric.data import Dataset

#numpy
import numpy as np

#scikit-learn
from sklearn import datasets

# own functionality
from dataset_property_util import Dataset_Properties_Manager

# An enum used to encode events for time logging if desired
class TimeLoggingEvent(Enum):
    load_graph = 0
    gen_k_disk = 1
    gen_r_s_ring = 2
    floyd_warshall = 3
    SP_vector_computation = 4
    get_database_idx = 5
    Two_WL_computation = 6

# Superclass implementing some basic logging and property extraction functionality
# NOTE: This class should be treated as an abstract class and is not intended to be directly initialized
class Feature_Generator():

    # NOTE: dataset._data.x must NOT be set to None to ensure proper function and num_features must be at least 1. In the unlabeled case, x should simply be set to the constant 0 vector.
    # samples defines the indices of the vertices in the dataset that should be considered.
    def __init__(self, dataset: Dataset, node_pred: bool, samples: Optional[List[int]] | Optional[Tensor], absolute_path_prefix: str, dataset_write_path: str, dataset_write_filename: str, dataset_desc: str, use_editmask: bool, result_mmap_dest: str, editmask_mmap_dest: str, split_desc: Optional[str] = None, properties_path: Optional[str] = None, write_properties_root_path: Optional[str] = None, write_properties_filename: Optional[str] = None, idx_lookup_path: Optional[str] = None, write_idx_lookup_path: Optional[str] = None, write_idx_lookup_filename: Optional[str] = None):
        super().__init__()

        self.dataset = dataset
        self.absolute_path_prefix = absolute_path_prefix
        self.dataset_path = dataset_write_path
        self.dataset_filename = f'{dataset_write_filename}.svmlight'
        self.editmask_filename = f'{dataset_write_filename}_editmask.svmlight'

        self.dataset_desc = dataset_desc
        self.use_editmask = use_editmask
        self.split_desc = split_desc

        # Nature of the prediction task
        self.node_pred = node_pred

        # Create properties from the dataset and load them
        self.prop_manager = Dataset_Properties_Manager(absolute_path_prefix = self.absolute_path_prefix, properties_path = properties_path, dataset = self.dataset, node_pred = node_pred, write_properties_root_path = write_properties_root_path, write_properties_filename = write_properties_filename)
        properties_file_path = self.prop_manager.properties_file_path
        self.samples, self.graph_samples, lookup_path = self.prop_manager.initialize_idx_lookups(lookup_path = idx_lookup_path, samples = samples, write_lookup_root_path = write_idx_lookup_path, write_lookup_filename = write_idx_lookup_filename)

        # Variables that need to be initialized by the subclasses
        # The computed dataset
        self.shared_dataset_result_dtype = None
        self.shared_dataset_result_shape = None
        self.shared_dataset_result_mmap_dest = None

        # Variables that need to be initialized by the subclasses
        # An editmask for the computed dataset (utilised in cropping)
        self.shared_editmask_dtype = None
        self.shared_editmask_shape = None
        self.shared_editmask_mmap_dest = None

        # Variables that need to be initialized by the subclasses
        # String describing the computed features
        self.title_str = None

        # Used in dictionary that will be filled with times if time logging is enabled
        self.num_events = len(TimeLoggingEvent)

        # Metadata that should be collected
        # The feature generation should collect:
        # - dataset properties: description, task_desc, (split_desc)
        # - result properties: num_samples, feature_dim, size (in array and on disk), path (of the stored result file)
        # - editmask properties (if available): num_features_cropped, path (on disk)
        # - configuration/input parameters: num_processes, chunksize, buffersize, log_times, graph_mode, use_editmask

        # Initialize for readable order in the file
        self.metadata = {}
        self.metadata["path_prefix"] = self.absolute_path_prefix
        self.metadata["dataset_prop"] = {}
        self.metadata["dataset_prop"]["desc"] = dataset_desc
        if self.node_pred:
            self.metadata["dataset_prop"]["task"] = "node classification"
        else:
            self.metadata["dataset_prop"]["task"] = "graph classification"
        if split_desc is not None:
            self.metadata["dataset_prop"]["split_desc"] = split_desc
        self.metadata["dataset_prop"]["properties_file_path"] = properties_file_path
        self.metadata["dataset_prop"]["idx_lookup_path"] = lookup_path
        self.metadata["result_prop"] = {}
        self.metadata["result_prop"]["feature_desc"] = ""
        self.metadata["result_prop"]["feature_identifier"] = {}
        self.metadata["result_prop"]["feature_identifier"]["id"] = ""
        self.metadata["result_prop"]["num_samples"] = -1
        self.metadata["result_prop"]["feature_dim"] = -1
        self.metadata["result_prop"]["size"] = {}
        self.metadata["result_prop"]["size"]["array"] = -1
        self.metadata["result_prop"]["size"]["disk"] = -1
        self.metadata["result_prop"]["path"] = ""
        if use_editmask:
            self.metadata["editmask"] = {}
            self.metadata["editmask"]["num_features_cropped"] = -1
            self.metadata["editmask"]["path"] = ""
        self.metadata["times"] = {}
        self.metadata["times"]["time_per_iteration"] = {}
        self.metadata["times"]["time_per_iteration"]["avg"] = -1
        self.metadata["times"]["time_per_iteration"]["min"] = -1
        self.metadata["times"]["time_per_iteration"]["max"] = -1
        self.metadata["times"]["feature_comp_overall"] = -1
        self.metadata["times"]["write_on_disk"] = -1
        self.metadata["config"] = {}
        self.metadata["config"]["graph_mode"] = False
        self.metadata["config"]["use_editmask"] = False
        self.metadata["config"]["num_processes"] = -1
        self.metadata["config"]["chunksize"] = -1
        self.metadata["config"]["buffer_size"] = -1
        self.metadata["config"]["log_times"] = False

        atexit.register(self.delete_mmaps)
        
    # Helper method used to clean up remaining mmaps after successfull execution to save disk space
    def delete_mmaps(self):
        try:
            os.remove(self.shared_dataset_result_mmap_dest)
        except:
            pass

        try:
            os.remove(self.shared_editmask_mmap_dest)
        except:
            pass
    
    # NOTE: this method requires a fully edited vector, meaning it has to already include the graph_id and vertex_id of the specific feature vector
    def edit_database_result_by_index(self, database_idx: int, vector: np.array):
        
        shared_dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'r+', shape = self.shared_dataset_result_shape)
        shared_dataset_result[database_idx,:] = vector
        shared_dataset_result.flush()

    # NOTE: this method requires a fully edited vector, meaning it has to already include the graph_id and vertex_id of the specific feature vector
    # Works with an array of indices and an array of vectors
    def edit_database_result_by_indices(self, database_idx: np.array, vectors: np.array, editmask: Optional[np.array], count: Optional[int] = None):

        if count is None:
            # A complete batch should be stored
            shared_dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'r+', shape = self.shared_dataset_result_shape)
            shared_dataset_result[database_idx,:] = vectors[:]
            shared_dataset_result.flush()

            if self.use_editmask:
                # Edit the editmask in the same step
                editmask_mmap = np.memmap(self.shared_editmask_mmap_dest, dtype = self.shared_editmask_dtype, mode = 'r+', shape = self.shared_editmask_shape)
                editmask_mmap[:] = np.logical_or(editmask_mmap, editmask)
                editmask_mmap.flush()
        else:
            # Only the first count elements should be stored
            shared_dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'r+', shape = self.shared_dataset_result_shape)
            shared_dataset_result[database_idx[:count],:] = vectors[:count,:]
            shared_dataset_result.flush()

            if self.use_editmask:
                # Edit the editmask in the same step
                editmask_mmap = np.memmap(self.shared_editmask_mmap_dest, dtype = self.shared_editmask_dtype, mode = 'r+', shape = self.shared_editmask_shape)
                editmask_mmap[:] = np.logical_or(editmask_mmap, editmask)
                editmask_mmap.flush()


    # NOTE: This method has to be overwritten by any subclass
    def compute_feature(self, idx: int):
        raise NotImplementedError
    
    # Utilised for analysis of runtimes, might include significant memory and calculation time overhead
    def compute_feature_log_times(self, idx: int):
        raise NotImplementedError
    
    # NOTE: The chunksize needs to be reduced for graph mode, normal: 512(/1024), graph: 64-128. Higher values might cause memory usage of the parent process to spike, low values might cause memroy usage of the parent process to continually rise.
    # The chunksize can be estimated based on CPU usage for each process: if the parent process is at max usage, the chunksize should probably be higher since to reduce administrative overhead
    # Additionally, this might lead to constantly increasing memory usage of the parent process since it has to buffer results from the child processes (especially in graph mode). This parameter has a huge impact on performance.
    # This computation should be done on an SSD, a hard drive might not have sufficient data speeds to support the computation without a significant slow down.

    # generate features of all vertices and store them on disk
    # graph_mode specifies whether the graphs are scheduled as tasks or their respective vertices. This is relevant if parts of the calculation can be used for the whole graph (e.g. for vertex SP features)
    # twoWL is necessary in case of computing 2WL features as the processes cannot return finished feature vectors since these would be inconsistent across processes, thus the parent process needs to do some additional processing.
    # NOTE: specifying a chunksize greater than 1 might increase the lag (esp of the progress bar) but often times yields a much faster computation speed and is highly advised
    # NOTE: In graph mode, the vector_buffer_size has to be larger than the largest graph
    def generate_features(self, chunksize: int = 1, vector_buffer_size: int = 256, num_processes: int=1, comment: Optional[str]=None, log_times: bool = False, dump_times: bool = False, metadata_path: Optional[str] = None, metadata_filename: Optional[str] = None, time_summary_path: str = "", time_summary_filename: Optional[str] = None, graph_mode: bool = False):

        assert num_processes > 0
        assert self.samples is not None and len(self.samples) > 0

        # We cannot schedule via graphs if there is only one graph available
        if self.node_pred:
            graph_mode = False

        if graph_mode:
            samples = self.graph_samples
        else:
            samples = self.samples

        # Set metadata for this run
        self.metadata["config"]["graph_mode"] = graph_mode
        self.metadata["config"]["use_editmask"] = self.use_editmask
        self.metadata["config"]["num_processes"] = num_processes
        self.metadata["config"]["chunksize"] = chunksize
        self.metadata["config"]["buffer_size"] = vector_buffer_size
        self.metadata["config"]["log_times"] = log_times

        t0 = time.time()

        # create buffers to store intermediary results of the feature vector calculations in order to batch up I/O operations
        vector_buffer = np.zeros(shape = (vector_buffer_size, self.shared_dataset_result_shape[1]), dtype = self.shared_dataset_result_dtype)
        vector_buffer_count = 0
        index_buffer = np.zeros(shape = (vector_buffer_size), dtype = int)
        editmask_buffer = None
        if self.use_editmask:
            editmask_buffer = np.full(shape = self.shared_editmask_shape, dtype = self.shared_editmask_dtype, fill_value = False)

        pool = multiprocessing.Pool(processes = num_processes)

        completed = 0
        num_samples = len(samples)

        # Time logging
        min_time = float('inf')
        max_time = -1
        sum_time = 0

        if log_times:
            self.times = np.full(shape = (len(self.samples), self.num_events), dtype = np.float32, fill_value = -1)
            feature_compute_func = self.compute_feature_log_times
        else:
            feature_compute_func = self.compute_feature

        # Store the time results
        for res in tqdm.tqdm(pool.imap_unordered(feature_compute_func, samples, chunksize = chunksize), total = len(samples)):
            if graph_mode:
                # The tasks are scheduled for whole graphs instead of vertices
                graph_time = res[3]
                sum_time += graph_time
                if graph_time < min_time:
                    min_time = graph_time
                if graph_time > max_time:
                    max_time = graph_time

                # res is in the shape of database_idx: array of database indices edited, result: vectors for the given indices, editmask_res: editmask result of the whole graph
                num_vec = res[0].shape[0]
                
                # Test whether remaining buffer space is sufficient, otherwise write the buffer and start it again
                if (vector_buffer_count + num_vec) >= vector_buffer_size:
                    # space insufficient

                    # write current buffer
                    self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer, count = vector_buffer_count)
                    vector_buffer_count = 0
                
                # Write result in the buffer
                index_buffer[vector_buffer_count:vector_buffer_count + num_vec] = res[0]
                vector_buffer[vector_buffer_count:vector_buffer_count + num_vec,:] = res[1]
                vector_buffer_count += num_vec
                completed += num_vec
                if self.use_editmask:
                    editmask_buffer[:] = np.logical_or(editmask_buffer, res[2])
                
                if completed == len(self.samples):
                    # finished computation, write final results
                    self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer, count = vector_buffer_count)

                if log_times:
                    # Store the times
                    for v, row in enumerate(res[4]):
                        self.times[res[0][v],:] = row

            else:
                # Tasks are scheduled for every vertex, thus each result only concerns a single vertex
                feature_time = res[3]
                sum_time += feature_time
                if feature_time < min_time:
                    min_time = feature_time
                if feature_time > max_time:
                    max_time = feature_time

                # Since the writing of intermediary results is by far the most time intensive task, we try to write batches of 256 vectors in the main process
                # vector_res is in the shape of database_idx, vector
                index_buffer[vector_buffer_count] = res[0]
                vector_buffer[vector_buffer_count,:] = res[1]
                vector_buffer_count += 1
                completed += 1
                if self.use_editmask:
                    editmask_buffer[:] = np.logical_or(editmask_buffer, res[2])

                if vector_buffer_count == vector_buffer_size:
                    # Buffer is full after adding the current result
                    self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer)
                    vector_buffer_count = 0
                elif completed == num_samples:
                    # Remaining tasks completed successfully but are not yet stored
                    self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer, count = vector_buffer_count)

                if log_times:
                    self.times[res[0],:] = res[4][:]
        # else:
        #     completed = 0
        #     num_samples = len(samples)

        #     for res in tqdm.tqdm(pool.imap_unordered(self.compute_feature, samples, chunksize = chunksize), total = num_samples):
        #         if graph_mode:
        #             # res is in the shape of database_idx: array of database indices edited, result: vectors for the given indices, editmask_res: editmask result of the whole graph

        #             num_vec = res[0].shape[0]
        #             # Test whether remaining buffer space is sufficient, otherwise write the buffer and start it again
        #             if (vector_buffer_count + num_vec) >= vector_buffer_size:
        #                 # space insufficient

        #                 # write current buffer
        #                 self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer, count = vector_buffer_count)
        #                 vector_buffer_count = 0
                    
        #             # Write result in the buffer
        #             index_buffer[vector_buffer_count:vector_buffer_count + num_vec] = res[0]
        #             vector_buffer[vector_buffer_count:vector_buffer_count + num_vec,:] = res[1]
        #             vector_buffer_count += num_vec
        #             completed += num_vec
        #             editmask_buffer[:] = np.logical_or(editmask_buffer, res[2])
                    
        #             if completed == len(self.samples):
        #                 # finished computation, write final results
        #                 self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer, count = vector_buffer_count)
        #         else:
        #             # Since the writing of intermediary results is by far the most time intensive task, we try to write batches of vector_buffer_size vectors in the main process
        #             # vector_res is in the shape of database_idx, vector
        #             index_buffer[vector_buffer_count] = res[0]
        #             vector_buffer[vector_buffer_count,:] = res[1]
        #             vector_buffer_count += 1
        #             completed += 1
        #             editmask_buffer[:] = np.logical_or(editmask_buffer, res[2])

        #             if vector_buffer_count == vector_buffer_size:
        #                 self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer)
        #                 vector_buffer_count = 0
        #             elif completed == num_samples:
        #                 # Remaining tasks completed successfully but are not yet stored
        #                 self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer, count = vector_buffer_count)

        pool.close()
        pool.join()

        computation_time = time.time() - t0

        self.metadata["times"]["time_per_iteration"]["avg"] = float(sum_time)/len(samples)
        self.metadata["times"]["time_per_iteration"]["min"] = min_time
        self.metadata["times"]["time_per_iteration"]["max"] = max_time
        self.metadata["times"]["feature_comp_overall"] = computation_time

        #should only be called after the processes are joined again
        if comment is not None:
            comment = self.title_str + "\n" + comment + "\n"
        else:
            comment = self.title_str + f"\n"
        
        self.save_features(comment)

        if log_times:
            time_summary_path = osp.join(self.absolute_path_prefix, time_summary_path)
            if time_summary_path != "":
                self.calculate_time_summary(time_summary_path = time_summary_path, time_summary_filename = time_summary_filename)
                if dump_times:
                    self.dump_times(time_dump_path = time_summary_path)

        # Write metadata of the run into a file
        if metadata_path is not None:
            assert metadata_filename is not None

            path = osp.join(self.absolute_path_prefix, metadata_path)
            if not osp.exists(path):
                os.makedirs(path)

            p = osp.join(path, metadata_filename)
            if not osp.exists(p):
                open(p, 'w').close()  

            with open(p, "w") as file:
                file.write(json.dumps(self.metadata, indent=4))

    # # Analogous implementatio nto the normal generate_features function, put into its own function due to significant differences in post-processing
    # def generate_features_two_WL(self, chunksize: int = 1, vector_buffer_size: int = 256, num_processes: int=1, comment: Optional[str]=None, log_times: bool = False, dump_times: bool = False, time_summary_path: str = "", time_summary_filename: Optional[str] = None):
    #     assert num_processes > 0
    #     assert self.samples is not None and len(self.samples) > 0

    #     samples = self.samples

    #     # We cannot schedule via graphs if there is only one graph available
    #     start_time = time.time()

    #     # We store each result as a dictionary of type (database_vertex_id, (graph_id, vertex_id)) -> dict[color -> freq] (IN MEMORY)
    #     color_freq_buffer = {}
    #     unique_vals = set([])

    #     pool = multiprocessing.Pool(processes = num_processes)
    #     if log_times:
    #         self.times = np.full(shape = (len(self.samples), self.num_events), dtype = np.float32, fill_value = -1)

    #         num_samples = len(samples)

    #         # Store the time results
    #         for res in tqdm.tqdm(pool.imap_unordered(self.compute_feature_log_times, samples, chunksize = chunksize), total = len(self.samples)):
    #             # We store the result in the buffer dictionary
    #             color_freq_buffer[tuple([res[0], res[1]])] = res[2]
    #             unique_vals = unique_vals.union(set(res[2].keys()))

    #             self.times[res[0],:] = res[3][:,:]
    #     else:
    #         num_samples = len(samples)

    #         for res in tqdm.tqdm(pool.imap_unordered(self.compute_feature, samples, chunksize = chunksize), total = num_samples):
    #             # We store the result in the buffer dictionary
    #             color_freq_buffer[tuple([res[0], res[1]])] = res[2]
    #             unique_vals = unique_vals.union(set(res[2].keys()))

    #     pool.close()
    #     pool.join()

    #     # should only be called after the processes are joined again

    #     # We need to post process the dictionary buffer and convert the color frequencies to vectors

    #     # Ensure well defined order of the finished vectors:
    #     unique_vals = list(unique_vals)
    #     unique_vals.sort()

    #     # Create a np array containing the results of shape |V|x|C|+2 where |C| is the number of unique colors computed by 2-WL on the dataset
    #     result_database = np.full(shape = (len(samples), len(unique_vals) + 2), fill_value = -1, dtype = np.int64)

    #     # All the unique colors generated values are stored in unique_vals
    #     for v, color_freq in color_freq_buffer.items():
    #         res = np.full(shape = (1, len(unique_vals) + 2), fill_value = -1, dtype = np.int64)
    #         res[0, 0] = v[1][0]
    #         res[0, 1] = v[1][1]
    #         for color_idx, color in enumerate(unique_vals):
    #             if color in color_freq:
    #                 res[0, color_idx + 2] = color_freq[color]
    #             else:
    #                 res[0, color_idx + 2] = 0
            
    #         result_database[v[0],:] = res[0,:]

    #     computation_time = time.time() - start_time

    #     if comment is not None:
    #         comment = self.title_str + "\n" + comment + f"\nComputation time: {computation_time}\n"
    #     else:
    #         comment = self.title_str + f"\nComputation time: {computation_time}\n"
        
    #     self.save_features(comment, use_mmap = False, result_database = result_database)

    #     if log_times:
    #         if time_summary_path != "":
    #             self.calculate_time_summary(time_summary_path = time_summary_path, time_summary_filename = time_summary_filename)
    #             if dump_times:
    #                 self.dump_times(time_dump_path = time_summary_path)

    #NOTE: This function must only be called after joining parallelized processes to ensure correct function
    # Implements a simple cropping of the feature vectors 
    def save_features(self, comment: Optional[str], use_mmap: bool = True, result_database: Optional[np.array] = None):
        t0 = time.time()

        rel_dataset_path = osp.join(self.dataset_path, self.dataset_filename)

        dataset_path = osp.join(self.absolute_path_prefix, self.dataset_path)
        if not osp.exists(dataset_path):
            os.makedirs(dataset_path)

        dataset_path = osp.join(dataset_path, self.dataset_filename)
        if not osp.exists(dataset_path):
            open(dataset_path, 'w').close()
        
        if self.use_editmask:
            editmask_path = osp.join(self.dataset_path, self.editmask_filename)
            path = osp.join(self.absolute_path_prefix, editmask_path)
            if not osp.exists(path):
                open(path, 'w').close()

        if not use_mmap:
            assert result_database is not None

        if comment is None:
            comment = ""

        self.metadata["result_prop"]["path"] = rel_dataset_path

        if use_mmap:
            dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'r+', shape = self.shared_dataset_result_shape)
            self.metadata["result_prop"]["num_samples"] = self.shared_dataset_result_shape[0]
            self.metadata["result_prop"]["size"]["array"] = dataset_result.nbytes

            if self.use_editmask:
                editmask = np.memmap(self.shared_editmask_mmap_dest, dtype = self.shared_editmask_dtype, mode = 'r+', shape = self.shared_editmask_shape)
                # number of True entries in editmask
                num_true = np.count_nonzero(editmask)
                self.metadata["result_prop"]["feature_dim"] = num_true
                self.metadata["editmask"]["num_features_cropped"] = self.shared_dataset_result_shape[1] - num_true
                self.metadata["editmask"]["path"] = editmask_path

                datasets.dump_svmlight_file(X = dataset_result[:,editmask], y = np.zeros(len(self.samples)), f = dataset_path, comment = comment)
                
                editmask = editmask.reshape(1,-1).astype(np.int64)
                datasets.dump_svmlight_file(X = editmask[:,:], y = np.zeros(1), f = editmask_path, comment = comment)
            else:
                self.metadata["result_prop"]["feature_dim"] = self.shared_dataset_result_shape[1]
                datasets.dump_svmlight_file(X = dataset_result[:,:], y = np.zeros(len(self.samples)), f = dataset_path, comment = comment)
        else:
            datasets.dump_svmlight_file(X = result_database[:,:], y = np.zeros(len(self.samples)), f = dataset_path, comment = comment)

        self.metadata["result_prop"]["size"]["disk"] = os.stat(dataset_path).st_size

        write_time = time.time() - t0
        self.metadata["times"]["write_on_disk"] = write_time

    # Helper to evaluate time complexity of individual operations
    def log_time(self, event: TimeLoggingEvent, value: float):
        
        self.times[event.value] = value

    def calculate_time_summary(self, time_summary_path: str, time_summary_filename: Optional[str] = None):
        assert len(self.times) > 0
        if time_summary_filename is None:
            time_summary_filename = "time_summary.txt"

        summary_str_list = []

        summary_str_list.append(f"Time summary for {self.title_str}:\n\n")

        # calculate average times for each step (/event)
        # The keys of summary are the events and the values are tuples (implemented as list with 2 elements) of the number of times a time for the specified event has been stored and the sum of the stored times.
        # The number of times a time for an event has been stored is useful to detect potentially erronious computation
        summary = np.zeros(shape = (self.num_events), dtype = np.float32)
        count = np.zeros(shape = (self.num_events), dtype = np.int32)

        for t in self.times:
            mask = (t >= 0)
            count[mask] += 1
            summary[mask] += t[mask]

        for event in TimeLoggingEvent:
            if count[event.value] == 0:
                continue

            if count[event.value] != len(self.samples):
                summary_str_list.append(f"WARNING: a time for {event.name} has not been stored for every iteration\n")

            avg = summary[event.value] / count[event.value]
            summary_str_list.append(f"{event.name} has been computed in {avg}s on average over {count[event.value]} iterations\n")

        summary_str = "".join(summary_str_list)
        
        # Write the summary into a file
        if not osp.exists(time_summary_path):
            os.makedirs(time_summary_path)

        p = osp.join(time_summary_path, time_summary_filename)
        if not osp.exists(p):
            open(p, 'w').close()  

        with open(p, "w") as file:
            file.write(summary_str)

    # Can be used to dump a json file with all the stored times
    def dump_times(self, time_dump_path: str, time_dump_filename: Optional[str] = None):
        if time_dump_filename is None:
            time_dump_filename = "times_dump.json"

        dump = {}
        for idx in range(self.shared_dataset_result_shape[0]):
            dump[idx] = {}
            for event in TimeLoggingEvent:
                if self.times[idx, event.value] >= 0:
                    dump[idx][event.name] = float(self.times[idx, event.value])

        if not osp.exists(time_dump_path):
            os.makedirs(time_dump_path)

        p = osp.join(time_dump_path, time_dump_filename)
        if not osp.exists(p):
            open(p, 'w').close()  

        with open(p, "w") as file:
            file.write(json.dumps(dump, indent=4))
