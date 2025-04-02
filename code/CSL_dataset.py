# Implementing the conversion of the CSL dataset from https://github.com/PurdueMINDS/RelationalPooling/tree/master/Synthetic_Data as a pytorch geometric InMemoryDataset
# Additionaly handles datasplits for k-fold CV

# general imports
import os.path as osp

# pytorch
import torch
import pickle

# pytorch geometric
from torch_geometric.data import InMemoryDataset, Data, download_url
from torch_geometric.utils import to_edge_index, remove_self_loops

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
            x = torch.zeros(num_vertices, dtype = torch.int64)
            target_label = torch.tensor(target_labels[idx])

            data = Data(x = x, edge_index = edge_index, y = target_label)
            data.num_nodes = num_vertices
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    # Generates a train, validation data split for k-fold CV (utilised on synthetic datasets)
    # TODO: implement
    def gen_train_val_split(self):
        raise NotImplementedError

    # Writes a train, validation data split for k-fold CV into a file. Required if the vertex feature computation and partition GNN computation are done in seperate instances to ensure consistend data splits.
    def write_train_val_split(self):
        raise NotImplementedError