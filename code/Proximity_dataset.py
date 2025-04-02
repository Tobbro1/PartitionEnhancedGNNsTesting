# Implements the proximity datasets by https://openreview.net/attachment?id=mWzWvMxuFg1&name=PDF_file given in https://github.com/radoslav11/SP-MPNN/tree/main

# general imports
import os.path as osp
import json

# pytorch
import pickle

# pytorch geometric
from torch_geometric.data import InMemoryDataset

class ProximityDataset(InMemoryDataset):

    # root is the root directory in which the dataset should be saved
    def __init__(self, root: str, h: int, transform = None, pre_transform = None, pre_filter = None):

        assert h in [1, 3, 5, 8, 10]
        self.h = h

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load the dataset if already present, otherwise start download
        self.load(self.processed_paths[0])

    # Needs to be overwritten
    # A List of the files that needs to be found to skip download
    def raw_file_names(self):
        return ['data_list.pickle']
    
    # Needs to be overwritten
    # A List of the files that needs to be found to skip processing
    def processed_file_names(self):
        return ['data.pt']
    
    # Needs to be overwritten
    # Download should never be executed but instead be done using the method in vertex_partition_feature_embedding_main.py
    def download(self):
        raise NotImplementedError

    # Needs to be overwritten
    # The data list is already given
    def process(self):
        data_list = None

        # Loading of the graphs from disk
        raw_path = osp.join(self.raw_dir, 'data_list.pickle')
        with open(raw_path, 'rb') as f:
            data_list = pickle.load(file = f)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    # Reads the corresponding data splits from disk
    def get_data_splits(self):
        splits = None

        with open(osp.join(self.root, f"{self.h}-Prox", f"{self.h}-Prox_splits.json"), 'r') as f:
            splits = json.load(f)

        return splits
    
    # Parses a data split
    