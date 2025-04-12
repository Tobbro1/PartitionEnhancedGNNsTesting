# Includes the functions to start experiments
from typing import Optional, Tuple, Dict, List
from copy import copy

import os.path as osp
import numpy as np
import os
import time

import torch
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

from clustering import Clustering_Algorithm, Vertex_Partition_Clustering
import util
import gnn_utils
from CSL_dataset import CSL_Dataset
from Proximity_dataset import ProximityDataset
import constants
from partition_gnns import GNN_Manager, Partition_enhanced_GIN, Partition_enhanced_GCN

# Manages the experiments and provides functionality to run them
class Experiment_Manager():

    def __init__(self, root_path: str, seed: Optional[int] = None):
        super().__init__()

        self.root_path = root_path

        if seed is None:
            seed = constants.SEED
        util.initialize_random_seeds(seed)

        self.dataset_path = None
        self.dataset_str = None
        self.gnn = None # GNN_Manager
        
        self.normalize_features = False
        
        self.num_reruns = constants.num_reruns

        self.h = None # Used for h-Prox datasets

        # lists of values of hyperparameters for grid search
        self.k = []
        self.r = []
        self.s = []
        self.is_vertex_sp_features = False
        self.num_clusters = []
        self.lsa_dims = []
        self.min_cluster_sizes = []
        self.num_layers = []
        self.hidden_channels = []
        self.batch_sizes = []
        self.num_epochs = []
        self.lrs = []

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
        self.data["num_reruns"] = self.num_reruns

    # Initialize variables, set hyperparameter ranges that need to be evaluated
    # base_model: one of 'gin' or 'gcn'
    # dataset_str: one of 'CSL' or h-Prox for any integer h from [1,3,5,8,10]
    def setup_experiments(self, dataset_str: str, base_model: str = 'gin', k: Optional[List[int]] = None, r: Optional[List[int]] = None, s: Optional[List[int]] = None, is_vertex_sp_features: bool = False,
                          num_clusters: List[int] = None, lsa_dims: List[int] = None, min_cluster_sizes: List[int] = None, num_layers: List[int] = None,
                          hidden_channels: List[int] = None, batch_sizes: List[int] = None, num_epochs: List[int] = None, lrs: List[float] = None, normalize_features: bool = None):

        self.gnn = GNN_Manager()

        self.dataset_str = dataset_str
        self.k = k
        self.r = r
        self.s = s
        self.is_vertex_sp_features = is_vertex_sp_features
        self.num_clusters = num_clusters
        self.lsa_dims = lsa_dims
        self.min_cluster_sizes = min_cluster_sizes
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.batch_sizes = batch_sizes
        self.num_epochs = num_epochs
        self.lrs = lrs
        self.normalize_features = normalize_features

        if base_model == 'gin':
            # Set up the functions to generate GNNs
            self.load_classic_gnn = self.gnn.generate_classic_GIN_model
            self.load_enhanced_gnn = self.gnn.generate_partition_enhanced_GIN_model
        elif base_model == 'gcn':
            self.load_classic_gnn = self.gnn.generate_classic_GCN_model
            self.load_enhanced_gnn = self.gnn.generate_partition_enhanced_GCN_model
        else:
            raise ValueError('Invalid gnn string')

        if dataset_str == 'CSL':
            self.dataset_path = osp.join('data', 'CSL', 'CSL_dataset')
        elif dataset_str.endswith('-Prox'):
            self.h = int(dataset_str[0])
            assert self.h in [1,3,5,8,10]
            self.dataset_path = osp.join('data', 'Proximity', f'{self.h}-Prox')
        else:
            raise ValueError('Invalid dataset string')

        print('---   Experiment setup complete   ---')

    def run_experiment(self):
        pass

    # Schedules multiple experiments
    def run_experiments(self):
        
        t0 = time.time()

        print(f'---   Starting experiments on {self.dataset_str}   ---')

        data = {}
        data["loss_func"] = "cross_entropy"
        data["overall_time"] = -1.0
        data["res"] = {}

        loss_func = F.cross_entropy

        if self.dataset_str == 'CSL':
            dataset = CSL_Dataset(root = osp.join(self.root_path, self.dataset_path))
        elif self.dataset_str.endswith('-Prox'):
            dataset = ProximityDataset(root = osp.join(self.root_path, self.dataset_path), h = self.h)

        splits = dataset.gen_data_splits()

        # Optimize GNN hyperparameters without clustering first
        print('---   Optimizing classical GNN hyperparameters   ---')
        hyperparameter_opt_data, best_val_accs, best_test_accs, best_classic_n_layers, best_classic_n_hidden_channels, best_classic_s_batch, best_classic_n_epoch, best_classic_lr = self.run_gnn_hyperparameter_optimization(splits = splits, loss_func = loss_func, classic_gnn = True)

        val_accs = torch.tensor(best_val_accs)
        test_accs = torch.tensor(best_test_accs)

        data["classic_gnn_hyperparameter_opt"] = hyperparameter_opt_data

        data["classic_gnn_hyperparameter_opt"]["res"]["val_acc"]["mean"] = val_accs.mean().item()
        data["classic_gnn_hyperparameter_opt"]["res"]["val_acc"]["std"] = val_accs.std().item()

        data["classic_gnn_hyperparameter_opt"]["res"]["test_acc"]["mean"] = test_accs.mean().item()
        data["classic_gnn_hyperparameter_opt"]["res"]["test_acc"]["std"] = test_accs.std().item()

        print(f'---   Val acc: mean: {val_accs.mean().item()}; std: {val_accs.std().item()}   ---')
        print(f'---   Test acc: mean: {test_accs.mean().item()}; std: {test_accs.std().item()}   ---')

        print('---   Optimizing classical GNN hyperparameters complete   ---')
        
        # # Testing
        # best_classic_n_layers = 3
        # best_classic_n_hidden_channels = 32
        # best_classic_s_batch = 128
        # best_classic_n_epoch = 1000
        # best_classic_lr = 0.01

        # Optimize cluster hyperparameter
        clusterer = Vertex_Partition_Clustering(self.root_path)

        print('---   Optimizing clustering hyperparameters   ---')
        # We do not need the explicit best features since we utilise the paths of the best result instead to avoid re-computing the clusterings
        # The max_num_clusters attribute is used since the best clustering might have less than best_num_clusters cluster (due to the min_cluster_size parameter)
        cluster_hyperparameter_opt_data, best_val_accs, best_test_accs, best_features_path, best_clustering_paths, max_num_clusters, best_num_clusters, best_lsa_dim, best_min_cluster_size = self.run_enhanced_gnn_cluster_hyperparameter_optimization(clusterer = clusterer, n_layers = best_classic_n_layers, 
                                                                                                                                                       hidden_channels = best_classic_n_hidden_channels, s_batch = best_classic_s_batch,
                                                                                                                                                       n_epoch = best_classic_n_epoch, lr = best_classic_lr, splits = splits, loss_func = loss_func)

        val_accs = torch.tensor(best_val_accs)
        test_accs = torch.tensor(best_test_accs)

        data["clustering_hyperparameter_opt"] = cluster_hyperparameter_opt_data

        data["clustering_hyperparameter_opt"]["res"]["val_acc"]["mean"] = val_accs.mean().item()
        data["clustering_hyperparameter_opt"]["res"]["val_acc"]["std"] = val_accs.std().item()

        data["clustering_hyperparameter_opt"]["res"]["test_acc"]["mean"] = test_accs.mean().item()
        data["clustering_hyperparameter_opt"]["res"]["test_acc"]["std"] = test_accs.std().item()

        print(f'---   Val acc: mean: {val_accs.mean().item()}; std: {val_accs.std().item()}   ---')
        print(f'---   Test acc: mean: {test_accs.mean().item()}; std: {test_accs.std().item()}   ---')

        print('---   Optimizing clustering hyperparameters complete   ---')

        # # Testing
        # best_features_path = osp.join(self.dataset_path, 'results', 'vertex_SP_features')
        # best_clustering_paths = {}
        # for idx in range(5):
        #     best_clustering_paths [idx] = osp.join(best_features_path, 'cluster-gnn', f'{idx}-fold', '4-means_min6140-size_cluster_metadata.json')
        # max_num_clusters = 1

        print('---   Running hyperparameter optimization for final enhanced GNN   ---')
        # Re-train enhanced gnn hyperparameter using clustering hyperparameter
        enhanced_hyperparameter_opt_data, best_val_accs, best_test_accs, best_n_layers, best_n_hidden_channels, best_s_batch, best_n_epoch, best_lr = self.run_gnn_hyperparameter_optimization(splits = splits, loss_func = loss_func, classic_gnn = False, best_clustering_paths = best_clustering_paths, best_num_clusters = max_num_clusters, vertex_feature_path = best_features_path)

        data["enhanced_gnn_hyperparameter_opt"] = enhanced_hyperparameter_opt_data

        # Test the computed GNN

        val_accs = torch.tensor(best_val_accs)
        test_accs = torch.tensor(best_test_accs)

        data["res"]["test_acc"] = {}
        data["res"]["test_acc"]["mean"] = test_accs.mean().item()
        data["res"]["test_acc"]["std"] = test_accs.std().item()
        data["res"]["val_acc"] = {}
        data["res"]["val_acc"]["mean"] = val_accs.mean().item()
        data["res"]["val_acc"]["std"] = val_accs.std().item()
        
        data["res"]["hyperparameter"] = {}

        data["res"]["hyperparameter"]["best_num_layers"] = best_n_layers
        data["res"]["hyperparameter"]["best_num_hidden_channels"] = best_n_hidden_channels
        data["res"]["hyperparameter"]["best_batch_size"] = best_s_batch
        data["res"]["hyperparameter"]["best_num_epochs"] = best_n_epoch
        data["res"]["hyperparameter"]["best_lr"] = best_lr

        data["res"]["hyperparameter"]["best_num_clusters"] = best_num_clusters
        data["res"]["hyperparameter"]["max_num_clusters"] = max_num_clusters
        data["res"]["hyperparameter"]["best_lsa_dim"] = best_lsa_dim
        data["res"]["hyperparameter"]["best_min_cluster_size"] = best_min_cluster_size

        print(f'---   Val acc: mean: {val_accs.mean().item()}; std: {val_accs.std().item()}   ---')
        print(f'---   Test acc: mean: {test_accs.mean().item()}; std: {test_accs.std().item()}   ---')

        print('---   Running hyperparameter optimization for final enhanced GNN completed   ---')

        data["overall_time"] = time.time() - t0

        self.data["csl_experiment"] = data

    def run_gnn_hyperparameter_optimization(self, splits: Dict, loss_func, classic_gnn: bool = True, best_clustering_paths: Optional[Dict] = None, best_num_clusters: Optional[int] = None, vertex_feature_path: Optional[str] = None) -> Tuple[Dict, GNN_Manager, int, int, int, int, float]:
        data = {}

        data["num_experiments"] = (len(self.num_layers) * len(self.hidden_channels) * len(self.batch_sizes) * len(self.num_epochs) * len(self.lrs))

        # Will be overwritten if training enhanced gnn hyperparameters
        if self.dataset_str == 'CSL':
            dataset = CSL_Dataset(root = osp.join(self.root_path, self.dataset_path))
        elif self.dataset_str.endswith('-Prox'):
            dataset = ProximityDataset(root = osp.join(self.root_path, self.dataset_path), h = self.h)

        # The best hyperparameters
        data["res"] = {}
        data["res"]["best_avg_val_acc"] = -1.0
        data["res"]["best_experiment_idx"] = -1
        data["res"]["best_num_layers"] = -1
        data["res"]["best_hidden_channels"] = -1
        data["res"]["best_batch_size"] = -1
        data["res"]["best_num_epochs"] = -1
        data["res"]["best_lr"] = -1.0
        # Set in the parent method
        data["res"]["val_acc"] = {}
        data["res"]["val_acc"]["mean"] = -1.0
        data["res"]["val_acc"]["std"] = -1.0

        data["res"]["test_acc"] = {}
        data["res"]["test_acc"]["mean"] = -1.0
        data["res"]["test_acc"]["std"] = -1.0

        # Stores the data for a single run
        data["experiment_idx"] = {}

        # Stores the average time used for computing a split, a rerun and an epoch (averaged over all total runs (including all folds))
        data["times"] = {}
        data["times"]["experiment_avg"] = -1.0
        data["times"]["split_avg"] = -1.0
        data["times"]["rerun_avg"] = -1.0
        data["times"]["epoch_avg"] = -1.0

        # iterate over the hyperparameters for this step
        best_avg_val_acc = 0.0
        best_n_layers = 0
        best_n_hidden_channels = 0
        best_s_batch = 0
        best_n_epoch = 0
        best_lr = -1.0
        best_val_acc_experiment_idx = 0

        best_val_accs = []
        best_test_accs = []

        avg_experiment_time = 0.0
        avg_fold_time_overall = 0.0
        avg_rerun_time_overall = 0.0
        avg_epoch_time_overall = 0.0

        cur_experiment_idx = 0
        for n_layers in self.num_layers:
            for n_hidden_channels in self.hidden_channels:
                for s_batch in self.batch_sizes:
                    for n_epoch in self.num_epochs:
                        for lr in self.lrs:
                            experiment_start = time.time()


                            data["experiment_idx"][cur_experiment_idx] = {}
                            data["experiment_idx"][cur_experiment_idx]["avg_val_acc"] = -1.0

                            # stored data:
                            # avg val acc
                            # foreach fold: avg val acc, split props, rerun data
                            # times: per rerun: avg, per fold: avg, per_epoch: avg
                            # Generate model
                            if classic_gnn:
                                self.gnn.set_dataset_parameters(num_classes = dataset.num_classes, num_features = dataset.num_features, num_clusters = 0)
                            else:
                                self.gnn.set_dataset_parameters(num_classes = dataset.num_classes, num_features = dataset.num_features, num_clusters = best_num_clusters)

                            if classic_gnn:
                                self.load_classic_gnn(hidden_channels = n_hidden_channels, num_layers = n_layers, lr = lr)
                            else:
                                self.load_enhanced_gnn(hidden_channels = n_hidden_channels, num_layers = n_layers, lr = lr)

                            data["experiment_idx"][cur_experiment_idx]["model"] = self.gnn.get_metadata()

                            data["experiment_idx"][cur_experiment_idx]["splits"] = {}

                            # k-fold CV
                            # data_folds stores the data for a single fold
                            data_folds = {}
                            
                            avg_val_acc_overall = 0.0
                            avg_time_fold = 0.0
                            avg_time_rerun = 0.0
                            avg_time_epoch = 0.0

                            test_accs = []
                            val_accs = []

                            for idx, split_dict in splits.items():

                                fold_start = time.time()

                                data_folds[idx] = {}
                                data_folds[idx]["avg_val_acc_split"] = -1.0

                                test_indices = split_dict["test"]
                                train_indices = split_dict["train"]
                                val_indices = split_dict["val"]

                                if self.dataset_str == 'CSL':
                                    split_prop = { "desc" : f"{constants.num_k_fold}-fold_CV", "split_mode" : "CV", "num_samples" : { "train" : len(train_indices), "val" : len(val_indices), "test" : len(test_indices) }, "split_idx" : idx}
                                elif self.dataset_str.endswith("-Prox"):
                                    split_prop = { "desc" : f"fixed_{self.h}-Prox", "split_mode" : "fixed", "num_samples" : { "train" : len(train_indices), "val" : len(val_indices), "test" : len(test_indices) }, "split_idx" : idx}
                                data_folds[idx]["split_prop"] = split_prop

                                train_indices = torch.tensor(train_indices, dtype = torch.long)
                                val_indices = torch.tensor(val_indices, dtype = torch.long)
                                test_indices = torch.tensor(test_indices, dtype = torch.long)

                                # Enhance the dataset with clustering_ids (using the previously computed information regarding the folds)
                                if not classic_gnn:
                                    if self.dataset_str == 'CSL':
                                        dataset = CSL_Dataset(root = osp.join(self.root_path, self.dataset_path))
                                    elif self.dataset_str.endswith('-Prox'):
                                        dataset = ProximityDataset(root = osp.join(self.root_path, self.dataset_path), h = self.h)
                                    feature_metadata_path = osp.join(vertex_feature_path, 'metadata.json')
                                    dataset, _ = gnn_utils.include_cluster_id_feature_transform(dataset = dataset, absolute_path_prefix = self.root_path, feature_metadata_path = feature_metadata_path, cluster_metadata_path = best_clustering_paths[idx])

                                # Evaluate a single experiment given the parameters
                                avg_val_acc, rerun_data, avg_test_acc, avg_time_rerun_fold, avg_time_epoch_fold = self.run_gnn_hyperparameter_experiment_split(dataset = dataset, s_batch = s_batch, n_epoch = n_epoch, train_indices = train_indices, val_indices = val_indices, test_indices = test_indices, loss_func = loss_func)

                                test_accs.append(avg_test_acc)
                                val_accs.append(avg_val_acc)

                                # store data
                                data_folds[idx]["avg_val_acc_split"] = avg_val_acc
                                data_folds[idx]["rerun"] = rerun_data

                                avg_val_acc_overall += avg_val_acc
                                avg_time_rerun += avg_time_rerun_fold
                                avg_time_epoch += avg_time_epoch_fold

                                avg_time_fold += time.time() - fold_start
                            
                            avg_time_fold /= constants.num_k_fold
                            avg_val_acc_overall /= constants.num_k_fold
                            avg_time_rerun /= constants.num_k_fold
                            avg_time_epoch /= constants.num_k_fold

                            data["experiment_idx"][cur_experiment_idx]["avg_val_acc"] = avg_val_acc_overall

                            data["experiment_idx"][cur_experiment_idx]["splits"] = data_folds

                            # These hyperparameters yield the best results so far
                            if avg_val_acc_overall > best_avg_val_acc:
                                best_avg_val_acc = avg_val_acc_overall
                                best_n_layers = n_layers
                                best_n_hidden_channels = n_hidden_channels
                                best_s_batch = s_batch
                                best_n_epoch = n_epoch
                                best_lr = lr
                                best_val_acc_experiment_idx = cur_experiment_idx
                                best_val_accs = val_accs
                                best_test_accs = test_accs
                            
                            cur_experiment_idx += 1

                            avg_fold_time_overall += avg_time_fold
                            avg_rerun_time_overall += avg_time_rerun
                            avg_epoch_time_overall += avg_time_epoch
                            avg_experiment_time += time.time() - experiment_start

        # Store results
        data["res"]["best_avg_val_acc"] = best_avg_val_acc
        data["res"]["best_experiment_idx"] = best_val_acc_experiment_idx
        data["res"]["best_num_layers"] = best_n_layers
        data["res"]["best_hidden_channels"] = best_n_hidden_channels
        data["res"]["best_batch_size"] = best_s_batch
        data["res"]["best_num_epochs"] = best_n_epoch
        data["res"]["best_lr"] = best_lr

        num_experiments = (len(self.num_layers) * len(self.hidden_channels) * len(self.batch_sizes) * len(self.num_epochs) * len(self.lrs))
        avg_experiment_time /= num_experiments
        avg_fold_time_overall /= num_experiments
        avg_rerun_time_overall /= num_experiments
        avg_epoch_time_overall /= num_experiments

        data["times"]["experiment_avg"] = avg_experiment_time
        data["times"]["split_avg"] = avg_fold_time_overall
        data["times"]["rerun_avg"] = avg_rerun_time_overall
        data["times"]["epoch_avg"] = avg_epoch_time_overall

        return data, best_val_accs, best_test_accs, best_n_layers, best_n_hidden_channels, best_s_batch, best_n_epoch, best_lr

    # Runs the gnn on one fold for given parameters and reports the result
    def run_gnn_hyperparameter_experiment_split(self, dataset: Dataset, s_batch: int, n_epoch: int, train_indices: Tensor, val_indices: Tensor, test_indices: Tensor, loss_func) -> Tuple[float, Dict, float, float, float]:

        # stored data format:
        # foreach rerun: val acc: best, epoch idx of best; epoch data
        # foreach epoch: val loss, train_loss, time
        
        rerun_data = {}

        avg_val_acc = 0.0
        avg_time_rerun = 0.0
        avg_time_epoch_overall = 0.0
        avg_test_acc = 0.0
        for cur_rerun in range(self.num_reruns):

            start_rerun = time.time()

            rerun_data[cur_rerun] = {}
            rerun_data[cur_rerun]["best_val_acc"] = -1.0
            rerun_data[cur_rerun]["best_val_acc_epoch_idx"] = -1
            rerun_data[cur_rerun]["epoch"] = {}

            self.gnn.model.reset_parameters()

            # We shuffle the training data for training
            train_loader = DataLoader(dataset = dataset[train_indices], batch_size = s_batch, shuffle = True)
            val_loader = DataLoader(dataset = dataset[val_indices], batch_size = s_batch, shuffle = False)
            test_loader = DataLoader(dataset = dataset[test_indices], batch_size = s_batch, shuffle = False)

            best_val_loss = float("inf")
            best_val_acc = 0.0
            best_val_epoch = 0
            avg_time_epoch = 0.0
            test_acc = -1.0
            for epoch in range(n_epoch):
                t0 = time.time()

                rerun_data[cur_rerun]["epoch"][epoch] = {}

                train_loss = self.train(gnn = self.gnn, loader = train_loader, loss_func = loss_func)
                val_loss = self.val(gnn = self.gnn, loader = val_loader, loss_func = loss_func)

                rerun_data[cur_rerun]["epoch"][epoch]["val_loss"] = val_loss
                rerun_data[cur_rerun]["epoch"][epoch]["train_loss"] = train_loss

                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    best_val_acc = self.test(gnn = self.gnn, loader = val_loader)
                    best_val_epoch = epoch
                    test_acc = self.test(gnn = self.gnn, loader = test_loader)

                time_epoch = time.time() - t0
                rerun_data[cur_rerun]["epoch"][epoch]["time"] = time_epoch
                avg_time_epoch += time_epoch
            
            avg_time_epoch /= n_epoch
            avg_time_epoch_overall += avg_time_epoch
            avg_test_acc += test_acc

            rerun_data[cur_rerun]["best_val_acc"] = best_val_acc
            rerun_data[cur_rerun]["best_val_acc_epoch_idx"] = best_val_epoch

            avg_val_acc += best_val_acc
            avg_time_rerun += time.time() - start_rerun

        avg_val_acc /= self.num_reruns
        avg_time_rerun /= self.num_reruns
        avg_time_epoch_overall /= self.num_reruns
        avg_test_acc /= self.num_reruns

        return avg_val_acc, rerun_data, avg_test_acc, avg_time_rerun, avg_time_epoch_overall
    
    # Optimizing the hyperparameters used for clustering: num_clusters, lsa_dim, min_cluster_size, k/r&s
    def run_enhanced_gnn_cluster_hyperparameter_optimization(self, clusterer: Vertex_Partition_Clustering, n_layers: int, hidden_channels: int, s_batch: int, n_epoch: int, lr: float, splits: Dict, loss_func) -> Tuple[Dict, str, Dict, int, int, int]:

        data = {}
        num_experiments = -1

        # A list of paths to all the vertex feature directories that should be considered for hyperparameter optimization
        vertex_feature_paths = []

        # decide whether k-disks, r-s-rings or vertex_sp_features are used.
        if self.k is not None and len(self.k) > 0:
            # Use k-disks
            num_experiments = len(self.num_clusters) * len(self.lsa_dims) * len(self.min_cluster_sizes) * len(self.k)
            for k in self.k:
                vertex_feature_paths.append(osp.join(self.dataset_path, 'results', f'{k}-disk_SP_features'))

        elif self.r is not None and self.s is not None and len(self.r) > 0 and len(self.s) == len(self.r):
            # Use r-s-rings
            num_experiments = len(self.num_clusters) * len(self.lsa_dims) * len(self.min_cluster_sizes) * len(self.r)
            for idx in range(len(self.r)):
                vertex_feature_paths.append(osp.join(self.dataset_path, 'results', f'{self.r[idx]}-{self.s[idx]}-ring_SP_features'))

        elif self.is_vertex_sp_features:
            num_experiments = len(self.num_clusters) * len(self.lsa_dims) * len(self.min_cluster_sizes)
            vertex_feature_paths.append(osp.join(self.dataset_path, 'results', f'vertex_SP_features'))

        else:
            raise ValueError("Invalid vertex feature identifiers")
        
        data["num_experiments"] = num_experiments

        data["model"] = {}

        # The best hyperparameters
        data["res"] = {}
        data["res"]["best_avg_val_acc"] = -1.0
        data["res"]["best_experiment_idx"] = -1
        data["res"]["best_features_path"] = ""
        data["res"]["best_num_clusters"] = -1
        data["res"]["best_lsa_dim"] = -1
        data["res"]["best_min_cluster_size"] = -1

        data["res"]["val_acc"] = {}
        data["res"]["val_acc"]["mean"] = -1.0
        data["res"]["val_acc"]["std"] = -1.0

        data["res"]["test_acc"] = {}
        data["res"]["test_acc"]["mean"] = -1.0
        data["res"]["test_acc"]["std"] = -1.0

        # Stores the data for a single run
        data["experiment_idx"] = {}
        
        # Stores the average time used for computing a fold, a rerun and an epoch (averaged over all total runs (including all folds))
        data["times"] = {}
        data["times"]["experiment_avg"] = -1.0
        data["times"]["cluster_avg"] = -1.0
        data["times"]["enhance_cluster_id_avg"] = -1.0
        data["times"]["split_avg"] = -1.0
        data["times"]["rerun_avg"] = -1.0
        data["times"]["epoch_avg"] = -1.0

        # iterate over the hyperparameters for this step
        best_avg_val_acc = 0.0
        best_features_path = ""
        best_clustering_metadata_paths = {} # For each experiment we store the cluster results for each fold
        best_num_clusters = 0
        best_lsa_dim = 0
        best_min_cluster_size = 0
        best_val_acc_experiment_idx = 0
        best_max_num_clusters = 0

        best_val_accs = []
        best_test_accs = []

        avg_experiment_time = 0.0
        avg_fold_time_overall = 0.0
        avg_rerun_time_overall = 0.0
        avg_epoch_time_overall = 0.0
        avg_time_cluster_overall = 0.0
        avg_time_enhance_cluster_id_overall = 0.0

        cur_experiment_idx = 0
        
        for n_cluster in self.num_clusters:
            for lsa_d in self.lsa_dims:
                for min_cluster_size in self.min_cluster_sizes:
                    for vertex_feature_path in vertex_feature_paths:
                        # Run the optimization, the different feature datasets are defined by the vertex_feature_path
                        experiment_start = time.time()

                        if self.dataset_str == 'CSL':
                            dataset = CSL_Dataset(root = osp.join(self.root_path, self.dataset_path))
                        elif self.dataset_str.endswith('-Prox'):
                            dataset = ProximityDataset(root = osp.join(self.root_path, self.dataset_path), h = self.h)

                        vertex_feature_metadata = util.read_metadata_file(osp.join(self.root_path, vertex_feature_path, 'metadata.json'))

                        data["experiment_idx"][cur_experiment_idx] = {}
                        data["experiment_idx"][cur_experiment_idx]["avg_val_acc"] = -1.0
                        data["experiment_idx"][cur_experiment_idx]["vertex_feature_path"] = vertex_feature_path

                        data["experiment_idx"][cur_experiment_idx]["model"] = {}
                        data["experiment_idx"][cur_experiment_idx]["splits"] = {}

                        # k-fold CV
                        # data_folds stores the data for a single fold
                        data_folds = {}
                        
                        avg_val_acc = 0.0
                        avg_time_cluster = 0.0
                        avg_time_enhance_cluster_id = 0.0
                        avg_time_fold = 0.0
                        avg_time_rerun = 0.0
                        avg_time_epoch = 0.0
                        fold_clustering_metadata_paths = {}
                        max_num_clusters = 0

                        test_accs = []
                        val_accs = []

                        # Initialize the gnn and dataset parameters
                        self.gnn.set_dataset_parameters(num_features = dataset.num_features, num_classes = dataset.num_classes, num_clusters = n_cluster)
                        self.load_enhanced_gnn(hidden_channels = hidden_channels, num_layers = n_layers, lr = lr)
                        data["experiment_idx"][cur_experiment_idx]["model"] = self.gnn.get_metadata()

                        for idx, split_dict in splits.items():
                            fold_start = time.time()
                            # Cluster the data based on the hyperparameters

                            data_folds[idx] = {}
                            data_folds[idx]["avg_val_acc_split"] = -1.0

                            test_indices = split_dict["test"]
                            train_indices = split_dict["train"]
                            val_indices = split_dict["val"]

                            if self.dataset_str == 'CSL':
                                split_prop = { "desc" : f"{constants.num_k_fold}-fold_CV", "split_mode" : "CV", "num_samples" : { "train" : len(train_indices), "val" : len(val_indices), "test" : len(test_indices) }, "split_idx" : idx}
                            elif self.dataset_str.endswith('-Prox'):
                                split_prop = { "desc" : f"fixed_{self.h}-Prox", "split_mode" : "fixed", "num_samples" : { "train" : len(train_indices), "val" : len(val_indices), "test" : len(test_indices) }, "split_idx" : idx}

                            data_folds[idx]["split_prop"] = split_prop

                            train_indices = np.array(train_indices, dtype = np.int32)
                            val_indices = np.array(val_indices, dtype = np.int32)

                            # Load the current dataset into the clusterer
                            clusterer.reset_parameters_and_metadata()
                            feature_vector_database_path = vertex_feature_metadata["result_prop"]["path"]
                            dataset_desc = vertex_feature_metadata["dataset_prop"]["desc"]
                            clusterer.load_dataset_from_svmlight(path = feature_vector_database_path, dtype = 'float64', dataset_desc = dataset_desc, split_prop = split_prop, normalize = self.normalize_features)

                            # LSA
                            working_path = osp.join(vertex_feature_path, 'cluster-gnn', f'{idx}-split')

                            if lsa_d > 0 and lsa_d < clusterer.dataset.shape[1]:
                                lsa_filename = f'{lsa_d}_dim_lsa.pkl'
                                clusterer.generate_lsa(target_dimensions = lsa_d, write_lsa_path = working_path, write_lsa_filename = lsa_filename)
                                clusterer.apply_lsa_to_dataset()

                            # k-means
                            if lsa_d > 0 and lsa_d < clusterer.dataset.shape[1]:
                                centroids_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_{lsa_d}-lsa_centroids.txt'
                            else:
                                centroids_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_centroids.txt'

                            _, centroids, _, clustering_time = clusterer.mini_batch_k_means(n_clusters = n_cluster, min_cluster_size = min_cluster_size, batch_size = constants.mbk_batch_size, n_init = constants.mbk_n_init, max_no_improvement = constants.mbk_max_no_improvement, max_iter = constants.mbk_max_iter)

                            clusterer.write_centroids_or_medoids(points = centroids, path = working_path, filename = centroids_filename)
                            avg_time_cluster += clustering_time

                            num_clusters = centroids.shape[0]
                            if num_clusters > max_num_clusters:
                                max_num_clusters = num_clusters

                            if lsa_d > 0 and lsa_d < clusterer.dataset.shape[1]:
                                metadata_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_{lsa_d}-lsa_cluster_metadata.json'
                            else:
                                metadata_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_cluster_metadata.json'

                            clusterer.write_metadata(path = working_path, filename = metadata_filename)
                            cluster_metadata = clusterer.metadata
                            cluster_metadata_path = osp.join(working_path, metadata_filename)

                            fold_clustering_metadata_paths[idx] = cluster_metadata_path

                            data_folds[idx]["cluster_metadata_path"] = cluster_metadata_path

                            # Create a new dataset enhanced with cluster_ids for GNN training
                            if self.dataset_str == 'CSL':
                                dataset = CSL_Dataset(root = osp.join(self.root_path, self.dataset_path))
                            elif self.dataset_str.endswith('-Prox'):
                                dataset = ProximityDataset(root = osp.join(self.root_path, self.dataset_path), h = self.h)
                            dataset, add_cluster_id_time = gnn_utils.include_cluster_id_feature_transform(absolute_path_prefix = self.root_path, dataset = dataset, feature_metadata = vertex_feature_metadata, cluster_metadata = cluster_metadata)
                            avg_time_enhance_cluster_id += add_cluster_id_time

                            # Convert the indices to tensors
                            train_indices = torch.from_numpy(train_indices).type(dtype = torch.long)
                            val_indices = torch.from_numpy(val_indices).type(dtype = torch.long)
                            test_indices = torch.tensor(test_indices, dtype = torch.long)

                            # Evaluate a single experiment given the parameters
                            avg_val_acc_fold, rerun_data, avg_test_acc_fold, avg_time_rerun_fold, avg_time_epoch_fold = self.run_gnn_hyperparameter_experiment_split(dataset = dataset, s_batch = s_batch, n_epoch = n_epoch, train_indices = train_indices, val_indices = val_indices, test_indices = test_indices, loss_func = loss_func)

                            # store data
                            data_folds[idx]["avg_val_acc_split"] = avg_val_acc_fold
                            data_folds[idx]["rerun"] = rerun_data

                            val_accs.append(avg_val_acc_fold)
                            test_accs.append(avg_test_acc_fold)

                            avg_val_acc += avg_val_acc_fold
                            avg_time_rerun += avg_time_rerun_fold
                            avg_time_epoch += avg_time_epoch_fold

                            avg_time_fold += time.time() - fold_start
                            
                        avg_time_fold /= constants.num_k_fold
                        avg_val_acc /= constants.num_k_fold
                        avg_time_rerun /= constants.num_k_fold
                        avg_time_epoch /= constants.num_k_fold
                        avg_time_cluster /= constants.num_k_fold
                        avg_time_enhance_cluster_id /= constants.num_k_fold

                        data["experiment_idx"][cur_experiment_idx]["avg_val_acc"] = avg_val_acc

                        data["experiment_idx"][cur_experiment_idx]["splits"] = data_folds

                        # These hyperparameters yield the best results so far
                        if avg_val_acc > best_avg_val_acc:
                            best_avg_val_acc = avg_val_acc
                            best_min_cluster_size = min_cluster_size
                            best_lsa_dim = lsa_d
                            best_num_clusters = n_cluster
                            best_features_path = vertex_feature_path
                            best_val_acc_experiment_idx = cur_experiment_idx
                            best_clustering_metadata_paths = fold_clustering_metadata_paths
                            best_max_num_clusters = max_num_clusters
                            best_val_accs = val_accs
                            best_test_accs = test_accs
                            
                        cur_experiment_idx += 1

                        avg_fold_time_overall += avg_time_fold
                        avg_rerun_time_overall += avg_time_rerun
                        avg_epoch_time_overall += avg_time_epoch
                        avg_time_cluster_overall += avg_time_cluster
                        avg_time_enhance_cluster_id_overall += avg_time_enhance_cluster_id

                        avg_experiment_time += time.time() - experiment_start

        # Store results
        data["res"]["best_avg_val_acc"] = best_avg_val_acc
        data["res"]["best_experiment_idx"] = best_val_acc_experiment_idx
        data["res"]["best_features_path"] = best_features_path
        data["res"]["best_num_clusters"] = best_num_clusters
        data["res"]["best_lsa_dim"] = best_lsa_dim
        data["res"]["best_min_cluster_size"] = best_min_cluster_size
        data["res"]["max_num_clusters"] = best_max_num_clusters

        avg_experiment_time /= num_experiments
        avg_fold_time_overall /= num_experiments
        avg_rerun_time_overall /= num_experiments
        avg_epoch_time_overall /= num_experiments
        avg_time_cluster_overall /= num_experiments
        avg_time_enhance_cluster_id_overall /= num_experiments

        data["times"]["experiment_avg"] = avg_experiment_time
        data["times"]["cluster_avg"] = avg_time_cluster_overall
        data["times"]["enhance_cluster_id_avg"] = avg_time_enhance_cluster_id_overall
        data["times"]["split_avg"] = avg_fold_time_overall
        data["times"]["rerun_avg"] = avg_rerun_time_overall
        data["times"]["epoch_avg"] = avg_epoch_time_overall

        return data, best_val_accs, best_test_accs, best_features_path, best_clustering_metadata_paths, best_max_num_clusters, best_num_clusters, best_lsa_dim, best_min_cluster_size

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
        gnn_generator.set_dataset_parameters(dataset = dataset_csl, num_clusters = num_clusters)

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
