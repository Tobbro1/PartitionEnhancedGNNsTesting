# Includes the functions to start experiments
from typing import Optional, Tuple

import os.path as osp
import numpy as np
import os

import torch

import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from clustering import Clustering_Algorithm, Vertex_Partition_Clustering
import util
import gnn_utils
from CSL_dataset import CSL_Dataset
import constants
from partition_gnns import GNN_Manager, Partition_enhanced_GIN, Partition_enhanced_GCN

# Manages the experiments and provides functionality to run them
class Experiment_Manager():

    def __init__(self, root_path: str, seed: Optional[int]):
        super().__init__()

        self.root_path = root_path

        if seed is None:
            seed = constants.SEED
        util.initialize_random_seeds(seed)

        self.dataset = None
        self.gnn = None # GNN_Manager
        
        self.num_reruns = constants.num_reruns

        # lists of values of hyperparameters for grid search
        k = []
        r = []
        s = []
        num_clusters = []
        lsa_dim = []
        num_layers = []
        hidden_channels = []
        batch_size = []
        num_epochs = []
        lr = []

        # Note for hyperparameter optimization: first the gnn props are trained without clustering (num_layers, hidden_channels, batch_size, num_epochs, lr)
        #                                       then, the clustering parameters are trained with the learned gnn parameters (k, r, s, num_clusters, lsa_dim)

        # Data collected when runninng the experiments
        # collects:
        # random seed
        # model data
        # config
        # dataset: desc, path (of dataset props and feature_vector metadata)
        # split method (CV/fixed), optional split desc
        # size of train, val, test splits (absolute, ratio)
        # for each run: config (incl loss func); clustering: desc, metadata path
        #               foreach epoch: loss (train, val, test), train_performance + std, test_performance + std, exec time
        # results: best_model parameters, times
        self.data = {}
        self.data["seed"] = seed
        self.data["model"] = {}
        self.data["config"] = {}
        self.data["num_reruns"] = self.num_reruns


    def setup_experiments(self, dataset_str: str, dataset_path: Optional[str] = None):
        # generate CV splits if necessary

        self.gnn = GNN_Manager()

        pass

    def run_experiment(self):
        pass

    # Schedules multiple experiments
    def run_csl_experiments(self, data_indices):
        
        loss_func = F.cross_entropy

        # Optimize GNN hyperparameters without clustering first

        splits = self.dataset.gen_train_val_split()

        for fold_idx in range(constants.num_k_fold):

            train_indices = splits[fold_idx]["train"]
            val_indices = splits[fold_idx]["val"]
            test_indices = splits["test"]

            split_prop = { "desc" : f"{constants.num_k_fold}-fold_CV", "split_mode" : "CV", "num_samples" : { "train" : len(train_indices), "val" : len(val_indices), "test" : len(val_indices) }, "split_idx" : fold_idx}

            train_indices = torch.tensor(train_indices, dtype = torch.long)
            val_indices = torch.tensor(val_indices, dtype = torch.long)
            # test_indices = torch.tensor(test_indices, dtype = torch.long)

            for _ in range(self.num_reruns):
                self.gnn.model.reset_parameters()

                avg_val_acc = 0.0
                


    # Trains the gnn based on loss_func, returns the average loss per graph
    def train(self, gnn: GNN_Manager, loader: DataLoader, loss_func) -> float:
        gnn.model.train()

        total_loss = 0.0
        for data in loader:
            data = data.to(gnn.device)
            gnn.optimizer.zero_grad()
            loss = loss_func(gnn.model(data), data.y)
            loss.backward()
            total_loss += data.num_graphs * loss.item()
            gnn.optimizer.step()

        return total_loss/len(loader.dataset)

    @torch.no_grad()
    def val(self, gnn: GNN_Manager, loader: DataLoader, loss_func) -> float:
        gnn.model.eval()

        total_loss = 0.0
        for data in loader:
            data = data.to(gnn.device)
            total_loss += data.num_graphs * loss_func(gnn.model(data), data.y).item()

        return total_loss / len(loader.dataset)
    
    @torch.no_grad()
    def test(self, gnn: GNN_Manager, loader: DataLoader) -> float:
        gnn.model.eval()

        correct = 0
        for data in loader:
            data = data.to(gnn.device)
            pred = gnn.model(data).max(dim = 1)[1]  # gnn.model(data).max(dim = 1) returns a tuple max_vals, max_indices
            correct += pred.eq(data.y).sum().item()

        return correct / len(loader.dataset)

    def run_experiment_old(self, root_path: str, working_path: str, vertex_feature_metadata_path: str):

        vertex_feature_metadata = util.read_metadata_file(path = osp.join(root_path, vertex_feature_metadata_path))

        # create a clustering with lsa
        clusterer = Vertex_Partition_Clustering(absolute_path_prefix = root_path)
        feature_vector_database_path = vertex_feature_metadata["result_prop"]["path"]
        dataset_desc = vertex_feature_metadata["dataset_prop"]["desc"]

        # split_prop is expected to be
        # ["desc"]
        # ["split_mode"]
        # ["num_samples"]
        # ["split_idx"] if split_mode is 'CV'
        split_desc = "Test_desc"
        split_mode = "Test"
        num_samples = -2
        split_idx = -1
        split_prop = { "desc" : split_desc, "split_mode" : split_mode, "num_samples" : num_samples, "split_idx" : split_idx}
        normalize = True

        clusterer.load_dataset_from_svmlight(path = feature_vector_database_path, dtype = 'float64', dataset_desc = dataset_desc, split_prop = split_prop, normalize = normalize)
        
        # LSA
        lsa_target_dim = 2
        lsa_filename = f'{lsa_target_dim}_dim_lsa.pkl'
        clusterer.generate_lsa(target_dimensions = lsa_target_dim, write_lsa_path = working_path, write_lsa_filename = lsa_filename)
        clusterer.apply_lsa_to_dataset()

        # k-means
        mbk_n_clusters = 6
        mbk_batch_size = 1024
        mbk_n_init = 10
        mbk_max_no_improvement = 10
        mbk_max_iter = 1000

        centroids_filename = f'{mbk_n_clusters}-means_centroids.txt'

        labels, centroids, inertia, clustering_time = clusterer.mini_batch_k_means(n_clusters = mbk_n_clusters, batch_size = mbk_batch_size, n_init = mbk_n_init, max_no_improvement = mbk_max_no_improvement, max_iter = mbk_max_iter)
        num_clusters = centroids.shape[0]

        clusterer.write_centroids_or_medoids(points = centroids, path = working_path, filename = centroids_filename)

        metadata_filename = 'cluster_metadata.json'
        clusterer.write_metadata(path = working_path, filename = metadata_filename)

        # TODO: Test gnn_utils.py
        csl_path = osp.join('data', 'CSL', 'CSL_dataset')
        dataset_csl = CSL_Dataset(root = osp.join(root_path, csl_path))

        feature_metadata_path = vertex_feature_metadata_path
        cluster_metadata_path = osp.join(working_path, metadata_filename)

        dataset_csl, add_cluster_id_time = gnn_utils.include_cluster_id_feature_transform(dataset = dataset_csl, absolute_path_prefix = root_path, feature_metadata_path = feature_metadata_path, cluster_metadata_path = cluster_metadata_path)
        # Since we added the cluster_ids

        gnn_generator = GNN_Manager()
        gnn_generator.load_dataset(dataset = dataset_csl, num_clusters = num_clusters)

        # GNN parameters
        batch_size = 128
        hidden_channels = 32
        num_layers = 3
        lr = 0.01
        epochs = 100
        # dropout

        gnn_generator.generate_partition_enhanced_GIN_model(hidden_channels = hidden_channels, num_layers = num_layers, lr = lr)
        train_loader = DataLoader(dataset = gnn_generator.dataset[:0.8], batch_size = batch_size, shuffle = False)

        loss = self.train_partition_enhanced_GNN(gnn = gnn_generator, train_loader = train_loader)

        print(loss)

    def train_partition_enhanced_GNN(self, gnn: GNN_Manager, train_loader: DataLoader) -> float:
        device = constants.device
        
        gnn.model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            gnn.optimizer.zero_grad()
            out = gnn.model(data)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            gnn.optimizer.step()
            total_loss += float(loss) * data.num_graphs

        return total_loss/len(train_loader.dataset)

if __name__ == '__main__':
    # test gnn util

    constants.initialize_random_seeds()

    torch.autograd.set_detect_anomaly(True)


    root_path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir)
    working_path = osp.join('data', 'CSL', 'CSL_dataset', 'results', 'vertex_sp_features')
    feature_metadata_path = osp.join(working_path, 'metadata.json')

    experiment_manager = Experiment_Manager()

    experiment_manager.run_experiment(root_path = root_path, working_path = osp.join(working_path, 'cluster-gnn'), vertex_feature_metadata_path = feature_metadata_path)
