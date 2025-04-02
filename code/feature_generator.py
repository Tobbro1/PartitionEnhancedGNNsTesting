#general imports
import multiprocessing.shared_memory
from typing import List, Optional, Tuple
import os
import os.path as osp
import time
import json #to implement saving and loading the properties of datasets without having to run the computation multiple times
import tqdm
from enum import Enum

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
    def __init__(self, dataset: Dataset, node_pred: bool, samples: Optional[List[int]] | Optional[Tensor], dataset_write_path: str, dataset_write_filename: str, result_mmap_dest: str, editmask_mmap_dest: str, properties_path: Optional[str] = None, write_properties_root_path: Optional[str] = None, write_properties_filename: Optional[str] = None):
        super().__init__()

        self.dataset = dataset
        self.dataset_path = dataset_write_path
        self.dataset_filename = dataset_write_filename

        self.properties = {}

        if properties_path is None:
            #compute general dataset properties => number of vertices, label alphabet, graph sizes; There may be further properties that need defining in the subclasses (such as a distance alphabet)
            self.properties["num_vertices"] = 0
            if dataset.num_features > 1:
                self.properties["label_alphabet"] = list(range(dataset.num_features))
            else:
                self.properties["label_alphabet"] = []

            self.properties["graph_sizes"] = []
            
            for i in range(dataset.len()):
                _curgraph_data = dataset.get(i)
                _curgraph_num_vertices = _curgraph_data.x.shape[0]
                self.properties["num_vertices"] += _curgraph_num_vertices
                self.properties["graph_sizes"].append(_curgraph_num_vertices)

                if dataset.num_features == 1:
                    _new_labels_tensor = torch.unique(input = _curgraph_data.x, sorted = False)
                    _new_labels = _new_labels_tensor.flatten().tolist()
                    self.properties["label_alphabet"].extend(_new_labels)
                    self.properties["label_alphabet"] = list(set(self.properties["label_alphabet"]))

            #if write_properties_path is specified, save properties as a json file on disk
            if write_properties_root_path is not None:
                assert write_properties_filename is not None

                if not osp.exists(write_properties_root_path):
                    os.makedirs(write_properties_root_path)

                p = osp.join(write_properties_root_path, write_properties_filename)
                if not osp.exists(p):
                    open(p, 'w').close()  

                with open(p, "w") as file:
                    file.write(json.dumps(self.properties, indent=4))
        else:
            #read properties file
            if not osp.exists(properties_path):
                raise FileNotFoundError
            else:
                with open(properties_path, "r") as file:
                    self.properties = json.loads(file.read())

        # Ensure important properties are set
        assert "num_vertices" in self.properties and "label_alphabet" in self.properties and "graph_sizes" in self.properties
        assert len(self.properties["graph_sizes"]) > 0 and len(self.properties["label_alphabet"]) > 0 and self.properties["num_vertices"] > 0

        # convert self.properties["graph_sizes"] to a numpy array for easier indexing later
        self.properties["graph_sizes"] = np.array(self.properties["graph_sizes"])

        # remove duplicates if necessary
        self.properties["label_alphabet"] = list(set(self.properties["label_alphabet"]))
        self.properties["label_alphabet"].sort()

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

        # Nature of the prediction task
        self.node_pred = node_pred

        # A dictionary utilised to speed up index computation methods. 
        # In the node prediction setting this corresponds to a map dataset_idx -> database_idx with entries for every dataset_idx contained in samples.
        # In the graph prediction setting this corresponds to a map dataset_idx -> graph_id, vertex_id with entries for every dataset_idx contained in the graphs in samples
        self.dataset_idx_lookup = {}

        # A dictionary containing key-value pairs of the form {graph_id: database_start_idx} giving the starting idx of vertices of the given graph in the database. 
        # This does not correspond to the idx in the dataset. This is used to efficiently compute a database_vertex_idx from a graph_id.
        # Only utilised in the graph prediction setting
        self.database_graph_start_idx = None

        if samples is None:
            # We compute the feature vectors for the whole dataset
            self.samples = []
            self.graph_samples = []

            if node_pred:
                # One graph
                for dataset_idx in range(self.properties["num_vertices"]):
                    self.dataset_idx_lookup[dataset_idx] = dataset_idx
                    self.samples.append(dataset_idx)
            else:
                # Multiple graphs
                self.database_graph_start_idx = {}

                self.graph_samples.append(0)
                graph_id = 0
                cur_start_idx = 0
                self.database_graph_start_idx[0] = 0
                for dataset_idx in range(self.properties["num_vertices"]):
                    if (dataset_idx - cur_start_idx) == int(self.properties["graph_sizes"][graph_id]):
                        graph_id += 1
                        self.graph_samples.append(graph_id)
                        cur_start_idx = dataset_idx
                        self.database_graph_start_idx[graph_id] = dataset_idx

                    self.dataset_idx_lookup[dataset_idx] = tuple([graph_id, dataset_idx - cur_start_idx])
                    self.samples.append(dataset_idx)
        else:
            if torch.is_tensor(samples):
                samples = samples.tolist()

            assert len(samples) > 0

            if node_pred:
                # One graph
                self.samples = []

                # Pre-compute the dataset_idx_lookup dictionary, a map representing dataset_idx -> database_idx
                for dataset_idx in samples:
                    self.dataset_idx_lookup[dataset_idx] = len(self.samples)
                    self.samples.append(dataset_idx)

            else:
                # Multiple graphs
                assert samples is not None and len(samples) > 0
                self.samples = []
                self.graph_samples = samples

                self.database_graph_start_idx = {}

                # We compute a list of the vertices included in the graphs (defined by the graph_ids in samples). Additionaly we store the start_idx property to efficiently compute the inverse of this operation via look-up.
                for graph_id in samples:
                    start_idx = int(sum(self.properties["graph_sizes"][:graph_id]))

                    # Store database_start_idx for a given graph. This is given by the length of the sample list before adding the vertices of the given graph.
                    assert graph_id not in self.database_graph_start_idx
                    self.database_graph_start_idx[graph_id] = len(self.samples)

                    # Refine samples
                    num_nodes = int(self.properties["graph_sizes"][graph_id])
                    self.samples.extend(list(range(start_idx, start_idx + num_nodes)))

                    # Pre-compute the dataset_idx_lookup dictionary
                    for dataset_idx in range(start_idx, start_idx + num_nodes):
                        self.dataset_idx_lookup[dataset_idx] = tuple([graph_id, dataset_idx - start_idx])

    
    # Helper to get the graph_id and vertex_id from a dataset idx. NOTE: The dataset idx is the idx used in the dataset, not in the tensor used to store results (which would be the database idx).
    def get_vertex_identifier_from_dataset_idx(self, dataset_idx: int) -> Tuple[int, int]:
        
        if self.node_pred:
            return tuple([0, dataset_idx])
        else:
            # We use a pre-computed dictionary representing dataset_idx -> graph_id, vertex_id to speed up the computation
            return self.dataset_idx_lookup[dataset_idx]
        
    # NOTE: In the node prediction scenario this method currently does not support computing a sample of the vertices of the dataset.
    def get_database_idx_from_vertex_identifier(self, vertex_identifier: Tuple[int,int]) -> int:   
    
        if self.node_pred:
            # Only one graph
            # vertex_identifier[1] is the dataset_idx of the given vertex. We use dataset_idx_lookup; a dictionary representing a map dataset_idx -> database_idx directly
            return self.dataset_idx_lookup[vertex_identifier[1]]
        else:
            # Multiple graphs, vertex_identifier[0] represents the graph_id and vertex_identifier[1] represents the idx of the vertex within the graph. Utilise graph_start_idx as a lookup method
            assert self.database_graph_start_idx is not None and vertex_identifier[0] in self.database_graph_start_idx

            return self.database_graph_start_idx[vertex_identifier[0]] + vertex_identifier[1]
    
    #NOTE: this method requires a fully edited vector, meaning it has to already include the graph_id and vertex_id of the specific feature vector
    def edit_database_result_by_index(self, database_idx: int, vector: np.array):
        
        shared_dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'r+', shape = self.shared_dataset_result_shape)
        shared_dataset_result[database_idx,:] = vector
        shared_dataset_result.flush()

    # NOTE: this method requires a fully edited vector, meaning it has to already include the graph_id and vertex_id of the specific feature vector
    # Works with an array of indices and an array of vectors
    def edit_database_result_by_indices(self, database_idx: np.array, vectors: np.array, editmask: np.array, count: Optional[int] = None):

        if count is None:
            # A complete batch should be stored
            shared_dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'r+', shape = self.shared_dataset_result_shape)
            shared_dataset_result[database_idx,:] = vectors[:]
            shared_dataset_result.flush()

            # Edit the editmask in the same step
            editmask_mmap = np.memmap(self.shared_editmask_mmap_dest, dtype = self.shared_editmask_dtype, mode = 'r+', shape = self.shared_editmask_shape)
            editmask_mmap[:] = np.logical_or(editmask_mmap, editmask)
            editmask_mmap.flush()
        else:
            # Only the first count elements should be stored
            shared_dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'r+', shape = self.shared_dataset_result_shape)
            shared_dataset_result[database_idx[:count],:] = vectors[:count,:]
            shared_dataset_result.flush()

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
    
    # generate features of all vertices and store them on disk
    # graph_mode specifies whether the graphs are scheduled as tasks or their respective vertices. This is relevant if parts of the calculation can be used for the whole graph (e.g. for vertex SP features)
    # twoWL is necessary in case of computing 2WL features as the processes cannot return finished feature vectors since these would be inconsistent across processes, thus the parent process needs to do some additional processing.
    # NOTE: specifying a chunksize greater than 1 might increase the lag (esp of the progress bar) but often times yields a much faster computation speed and is highly advised
    # NOTE: In graph mode, the vector_buffer_size has to be larger than the largest graph
    def generate_features(self, chunksize: int = 1, vector_buffer_size: int = 256, num_processes: int=1, comment: Optional[str]=None, log_times: bool = False, dump_times: bool = False, time_summary_path: str = "", time_summary_filename: Optional[str] = None, graph_mode: bool = False, two_WL: bool = False):

        if two_WL:
            return self.generate_features_two_WL(chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, comment = comment, log_times = log_times, dump_times = dump_times, time_summary_path = time_summary_path, time_summary_filename = time_summary_filename)

        assert num_processes > 0
        assert self.samples is not None and len(self.samples) > 0

        # We cannot schedule via graphs if there is only one graph available
        if self.node_pred:
            graph_mode = False

        if graph_mode:
            samples = self.graph_samples
        else:
            samples = self.samples

        start_time = time.time()

        vector_buffer = np.zeros(shape = (vector_buffer_size, self.shared_dataset_result_shape[1]), dtype = self.shared_dataset_result_dtype)
        vector_buffer_count = 0
        index_buffer = np.zeros(shape = (vector_buffer_size), dtype = int)
        editmask_buffer = np.full(shape = self.shared_editmask_shape, dtype = self.shared_editmask_dtype, fill_value = False)

        pool = multiprocessing.Pool(processes = num_processes)
        if log_times:
            self.times = np.full(shape = (len(self.samples), self.num_events), dtype = np.float32, fill_value = -1)

            completed = 0
            num_samples = len(samples)

            # Store the time results
            for res in tqdm.tqdm(pool.imap_unordered(self.compute_feature_log_times, samples, chunksize = chunksize), total = len(samples)):
                if graph_mode:
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
                    editmask_buffer[:] = np.logical_or(editmask_buffer, res[2])
                    
                    if completed == len(self.samples):
                        # finished computation, write final results
                        self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer, count = vector_buffer_count)

                    for v, row in enumerate(res[3]):
                        self.times[res[0][v],:] = row

                else:
                    # Since the writing of intermediary results is by far the most time intensive task, we try to write batches of 256 vectors in the main process
                    # vector_res is in the shape of database_idx, vector
                    index_buffer[vector_buffer_count] = res[0]
                    vector_buffer[vector_buffer_count,:] = res[1]
                    vector_buffer_count += 1
                    completed += 1
                    editmask_buffer[:] = np.logical_or(editmask_buffer, res[2])

                    if vector_buffer_count == vector_buffer_size:
                        self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer)
                        vector_buffer_count = 0
                    elif completed == num_samples:
                        # Remaining tasks completed successfully but are not yet stored
                        self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer, count = vector_buffer_count)

                    self.times[res[0],:] = res[3][:,:]
        else:
            completed = 0
            num_samples = len(samples)

            for res in tqdm.tqdm(pool.imap_unordered(self.compute_feature, samples, chunksize = chunksize), total = num_samples):
                if graph_mode:
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
                    editmask_buffer[:] = np.logical_or(editmask_buffer, res[2])
                    
                    if completed == len(self.samples):
                        # finished computation, write final results
                        self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer, count = vector_buffer_count)
                else:
                    # Since the writing of intermediary results is by far the most time intensive task, we try to write batches of vector_buffer_size vectors in the main process
                    # vector_res is in the shape of database_idx, vector
                    index_buffer[vector_buffer_count] = res[0]
                    vector_buffer[vector_buffer_count,:] = res[1]
                    vector_buffer_count += 1
                    completed += 1
                    editmask_buffer[:] = np.logical_or(editmask_buffer, res[2])

                    if vector_buffer_count == vector_buffer_size:
                        self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer)
                        vector_buffer_count = 0
                    elif completed == num_samples:
                        # Remaining tasks completed successfully but are not yet stored
                        self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer, count = vector_buffer_count)

        pool.close()
        pool.join()

        computation_time = time.time() - start_time

        #should only be called after the processes are joined again
        if comment is not None:
            comment = self.title_str + "\n" + comment + f"\nComputation time: {computation_time}\n"
        else:
            comment = self.title_str + f"\nComputation time: {computation_time}\n"
        
        self.save_features(comment)

        if log_times:
            if time_summary_path != "":
                self.calculate_time_summary(time_summary_path = time_summary_path, time_summary_filename = time_summary_filename)
                if dump_times:
                    self.dump_times(time_dump_path = time_summary_path)

    # Analogous implementatio nto the normal generate_features function, put into its own function due to significant differences in post-processing
    def generate_features_two_WL(self, chunksize: int = 1, vector_buffer_size: int = 256, num_processes: int=1, comment: Optional[str]=None, log_times: bool = False, dump_times: bool = False, time_summary_path: str = "", time_summary_filename: Optional[str] = None):
        assert num_processes > 0
        assert self.samples is not None and len(self.samples) > 0

        samples = self.samples

        # We cannot schedule via graphs if there is only one graph available
        start_time = time.time()

        # We store each result as a dictionary of type (database_vertex_id, (graph_id, vertex_id)) -> dict[color -> freq] (IN MEMORY)
        color_freq_buffer = {}
        unique_vals = set([])

        pool = multiprocessing.Pool(processes = num_processes)
        if log_times:
            self.times = np.full(shape = (len(self.samples), self.num_events), dtype = np.float32, fill_value = -1)

            num_samples = len(samples)

            # Store the time results
            for res in tqdm.tqdm(pool.imap_unordered(self.compute_feature_log_times, samples, chunksize = chunksize), total = len(self.samples)):
                # We store the result in the buffer dictionary
                color_freq_buffer[tuple([res[0], res[1]])] = res[2]
                unique_vals = unique_vals.union(set(res[2].keys()))

                self.times[res[0],:] = res[3][:,:]
        else:
            num_samples = len(samples)

            for res in tqdm.tqdm(pool.imap_unordered(self.compute_feature, samples, chunksize = chunksize), total = num_samples):
                # We store the result in the buffer dictionary
                color_freq_buffer[tuple([res[0], res[1]])] = res[2]
                unique_vals = unique_vals.union(set(res[2].keys()))

        pool.close()
        pool.join()

        # should only be called after the processes are joined again

        # We need to post process the dictionary buffer and convert the color frequencies to vectors

        # Ensure well defined order of the finished vectors:
        unique_vals = list(unique_vals)
        unique_vals.sort()

        # Create a np array containing the results of shape |V|x|C|+2 where |C| is the number of unique colors computed by 2-WL on the dataset
        result_database = np.full(shape = (len(samples), len(unique_vals) + 2), fill_value = -1, dtype = np.int64)

        # All the unique colors generated values are stored in unique_vals
        for v, color_freq in color_freq_buffer.items():
            res = np.full(shape = (1, len(unique_vals) + 2), fill_value = -1, dtype = np.int64)
            res[0, 0] = v[1][0]
            res[0, 1] = v[1][1]
            for color_idx, color in enumerate(unique_vals):
                if color in color_freq:
                    res[0, color_idx + 2] = color_freq[color]
                else:
                    res[0, color_idx + 2] = 0
            
            result_database[v[0],:] = res[0,:]

        computation_time = time.time() - start_time

        if comment is not None:
            comment = self.title_str + "\n" + comment + f"\nComputation time: {computation_time}\n"
        else:
            comment = self.title_str + f"\nComputation time: {computation_time}\n"
        
        self.save_features(comment, use_mmap = False, result_database = result_database)

        if log_times:
            if time_summary_path != "":
                self.calculate_time_summary(time_summary_path = time_summary_path, time_summary_filename = time_summary_filename)
                if dump_times:
                    self.dump_times(time_dump_path = time_summary_path)

    #NOTE: This function must only be called after joining parallelized processes to ensure correct function
    # Implements a simple cropping of the feature vectors 
    def save_features(self, comment: Optional[str], use_mmap: bool = True, result_database: Optional[np.array] = None):

        if not osp.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        p = osp.join(self.dataset_path, self.dataset_filename)
        if not osp.exists(p):
            open(p, 'w').close()

        if not use_mmap:
            assert result_database is not None

        if use_mmap:
            dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'r+', shape = self.shared_dataset_result_shape)
            editmask = np.memmap(self.shared_editmask_mmap_dest, dtype = self.shared_editmask_dtype, mode = 'r+', shape = self.shared_editmask_shape)

            if comment is None:
                datasets.dump_svmlight_file(X = dataset_result[:,editmask], y = np.zeros(len(self.samples)), f = p, comment = "")
            else:
                datasets.dump_svmlight_file(X = dataset_result[:,editmask], y = np.zeros(len(self.samples)), f = p, comment = comment)
        else:
            if comment is None:
                datasets.dump_svmlight_file(X = result_database[:,:], y = np.zeros(len(self.samples)), f = p, comment = "")
            else:
                datasets.dump_svmlight_file(X = result_database[:,:], y = np.zeros(len(self.samples)), f = p, comment = comment)

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
