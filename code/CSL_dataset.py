# Implementing the conversion of the CSL dataset from https://github.com/PurdueMINDS/RelationalPooling/tree/master/Synthetic_Data as a pytorch geometric InMemoryDataset
# Additionaly handles datasplits for k-fold CV

# general imports
from typing import Optional, List
import os.path as osp

import numpy as np

# pytorch
import torch
import pickle

# pytorch geometric
from torch_geometric.data import InMemoryDataset, Data, download_url
from torch_geometric.utils import to_edge_index, remove_self_loops

import util
import constants

class CSL_Dataset(InMemoryDataset):

    # root is the root directory in which the dataset should be saved
    def __init__(self, root: str, transform = None, pre_transform = None, pre_filter = None):
        super().__init__(root, transform, pre_transform, pre_filter)

        # Load the dataset if already present, otherwise start download
        self.load(self.processed_paths[0])

    # Needs to be overwritten
    # A List of the files that needs to be found to skip download
    def raw_file_names(self):
        return ['graphs_Kary_Deterministic_Graphs.pkl', 'y_Kary_Deterministic_Graphs.pt']
    
    # Needs to be overwritten
    # A List of the files that needs to be found to skip processing
    def processed_file_names(self):
        return ['csl_data.pt']
    
    # Needs to be overwritten
    # Download to self.raw_dir
    def download(self):
        graph_adjs_url = 'https://github.com/PurdueMINDS/RelationalPooling/raw/refs/heads/master/Synthetic_Data/graphs_Kary_Deterministic_Graphs.pkl'
        graph_targets_url = 'https://github.com/PurdueMINDS/RelationalPooling/raw/refs/heads/master/Synthetic_Data/y_Kary_Deterministic_Graphs.pt'

        download_url(graph_adjs_url, self.raw_dir)
        download_url(graph_targets_url, self.raw_dir)

    # Needs to be overwritten
    # Processes the data into a list of Data objects before collating them and saving them in the save method.
    def process(self):
        data_list = []

        # Loading of the graphs from disk
    	# sparse_adjs is a list of (150) adjacency matrices representing the graphs. We will need to convert them to a format consisting of an edge_index tensor and a label tensor (here: a consistent 0-tensor)
        sparse_adjs = None
        with open(osp.join(self.raw_dir, self.raw_file_names()[0]), 'rb') as f:
            sparse_adjs = pickle.load(file = f)

        target_labels = torch.load(osp.join(self.raw_dir, self.raw_file_names()[1]))

        # Processing of the graphs and converting them to pytorch geometric Data objects
        for idx, sparse_adj in enumerate(sparse_adjs):
            adj = torch.from_numpy(sparse_adj.toarray())
            num_vertices = adj.shape[0]
            # We remove self-loops from the graph. This should not impact the expressivity results and is more consistent with the definition.
            edge_index = remove_self_loops(edge_index = to_edge_index(adj = adj.to_sparse())[0])[0]
            x = torch.zeros((num_vertices, 1), dtype = torch.int64)
            target_label = torch.tensor(target_labels[idx])

            data = Data(x = x, edge_index = edge_index, y = target_label)
            data.num_nodes = num_vertices
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def get_classes(self) -> List[int]:
        return torch.unique(self._data.y).view(-1).tolist()

    # Generates a train, validation data split for k-fold CV (utilised on synthetic datasets)
    def gen_data_splits(self, path: Optional[str] = None, write_path: Optional[str] = None, write_filename: Optional[str] = None):
        
        splits = {}

        if path is not None:
            splits = util.read_metadata_file(path = path)
        else:
            # generate test data

            # We follow the split setup from https://proceedings.mlr.press/v97/murphy19a/murphy19a.pdf

            # We need to construct balanced splits based on class

            classes = self.get_classes()

            k = constants.num_k_fold

            for idx in range(k):
                splits[idx] = {}
                splits[idx]["test"] = [] # Note that the test set is the same for each fold, we store it multiple times to be consistent with the Prox Dataset splits
                splits[idx]["train"] = []
                splits[idx]["val"] = []

            # Index in which the larger validation set is put (if the size of a class is not divisable by k). Done to ensure roughly equal sizes of train/validation splits across folds.
            cur_unbalanced_idx = 0
            for c in classes:
                possible_class_elements = (self._data.y.view(-1) == c).nonzero().view(-1)
                possible_class_elements = np.array(possible_class_elements.tolist(), dtype = np.int64)

                num_graphs = possible_class_elements.shape[0]

                num_test_graphs = int(constants.k_fold_test_ratio * num_graphs)
                possible_indices = np.array(range(num_graphs), dtype = np.int64)
                test_indices = np.random.choice(possible_indices, size = num_test_graphs, replace = False)
                test_indices_list = possible_class_elements[test_indices].tolist()

                # generate k folds

                fold_elements = np.delete(possible_class_elements, test_indices)
                num_remaining_graphs = fold_elements.shape[0]
                num_select = int(num_remaining_graphs / k)

                remaining_elements = np.copy(fold_elements)

                for idx in range(k):
                    splits[idx]["test"].extend(test_indices_list)

                    if idx == cur_unbalanced_idx:
                        continue

                    val_elements = np.random.choice(remaining_elements, size = num_select, replace = False)
                    splits[idx]["train"].extend(np.delete(fold_elements, np.isin(fold_elements, val_elements)).tolist())
                    splits[idx]["val"].extend(val_elements.tolist())
                    remaining_elements = np.delete(remaining_elements, np.isin(remaining_elements, val_elements))

                # last iteration gets the remaining graphs
                idx = cur_unbalanced_idx
                val_elements = remaining_elements
                splits[idx]["train"].extend(np.delete(fold_elements, np.isin(fold_elements, val_elements)).tolist())
                splits[idx]["val"].extend(val_elements.tolist())

                cur_unbalanced_idx += 1
                cur_unbalanced_idx %= k

            # sort
            for idx in range(k):
                splits[idx]["train"].sort()
                splits[idx]["val"].sort()
                splits[idx]["test"].sort()

            if write_path is not None:
                self.write_train_val_split(splits = splits, path = write_path, filename = write_filename)

        return splits

    # Writes a train, validation data split for k-fold CV into a file. Required if the vertex feature computation and partition GNN computation are done in seperate instances to ensure consistend data splits.
    def write_train_val_split(self, splits, path: str, filename: str):
        util.write_metadata_file(path = path, filename = filename, data = splits)