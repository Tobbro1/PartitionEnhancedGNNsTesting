# Includes the functions to start experiments
from typing import Optional, Tuple, Dict, List
from copy import deepcopy

import os.path as osp
import numpy as np
import os
import time
from tqdm.auto import tqdm

import torch
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset

from clustering import Vertex_Partition_Clustering
import util
import gnn_utils
from CSL_dataset import CSL_Dataset
from Proximity_dataset import ProximityDataset
import constants
from partition_gnns import GNN_Manager
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

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
        self.model_str = None
        self.run_experiments = None

        self.criterion = None # used for ogb datasets
        
        self.is_lovasz_feature = False
        self.lo_idx_str = ""
        self.normalize_features = False
        
        self.num_reruns = constants.num_reruns

        self.h = None # Used for h-Prox datasets

        self.max_patience = float('inf')

        # lists of values of hyperparameters for grid search
        self.k = []
        self.r = []
        self.s = []
        self.is_vertex_sp_features = False
        self.num_clusters = []
        self.pca_dims = []
        self.min_cluster_sizes = []
        self.num_layers = []
        self.hidden_channels = []
        self.batch_sizes = []
        self.num_epochs = []
        self.lrs = []

        self.classical_gnn_hyperparameter_experiment_res_path = None
        self.clustering_hyperparameter_experiment_res_path = None

        self.base_model = ""

        self.use_gpnn = False
        self.gpnn_layers = []
        self.gpnn_channels = []

        # Note for hyperparameter optimization: first the gnn props are trained without clustering (num_layers, hidden_channels, batch_size, num_epochs, lr)
        #                                       then, the clustering parameters are trained with the learned gnn parameters (k, r, s, num_clusters, pca_dim)

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
    def setup_experiments(self, dataset_str: str, base_model: str = 'gin', use_gpnn: bool = False, gpnn_channels: List[int] = None, gpnn_layers: List[int] = None, is_lovasz_feature: bool = False, 
                          lo_idx_str: str = "", k: Optional[List[int]] = None, r: Optional[List[int]] = None, s: Optional[List[int]] = None, is_vertex_sp_features: bool = False,
                          num_clusters: List[int] = None, pca_dims: List[int] = None, min_cluster_sizes: List[int] = None, num_layers: List[int] = None,
                          hidden_channels: List[int] = None, batch_sizes: List[int] = None, num_epochs: List[int] = None, lrs: List[float] = None, normalize_features: bool = None, exp_mode: int = -1, max_patience: Optional[int] = None,
                          prev_res_path: Optional[str] = None):

        self.gnn = GNN_Manager()

        self.dataset_str = dataset_str
        self.k = k
        self.r = r
        self.s = s
        self.is_vertex_sp_features = is_vertex_sp_features
        self.num_clusters = num_clusters
        self.pca_dims = pca_dims
        self.min_cluster_sizes = min_cluster_sizes
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.batch_sizes = batch_sizes
        self.num_epochs = num_epochs
        self.lrs = lrs
        self.is_lovasz_feature = is_lovasz_feature
        self.lo_idx_str = lo_idx_str
        self.normalize_features = normalize_features
        self.base_model = base_model

        self.use_gpnn = use_gpnn
        self.gpnn_layers = gpnn_layers
        self.gpnn_channels = gpnn_channels

        if max_patience is not None:
            self.max_patience = max_patience

        if base_model == 'gin':
            # Set up the functions to generate GNNs
            self.load_classic_gnn = self.gnn.generate_classic_GIN_model
            if self.use_gpnn:
                self.load_enhanced_gnn = self.gnn.generate_GPNN_model
            else:
                self.load_enhanced_gnn = self.gnn.generate_partition_enhanced_GIN_model
        elif base_model == 'gcn':
            self.load_classic_gnn = self.gnn.generate_classic_GCN_model
            if self.use_gpnn:
                self.load_enhanced_gnn = self.gnn.generate_GPNN_model
            else:
                self.load_enhanced_gnn = self.gnn.generate_partition_enhanced_GCN_model
        else:
            raise ValueError('Invalid gnn string')
        
        self.model_str = base_model

        assert exp_mode > -1

        
        if dataset_str == 'ogbg-ppa':
            self.dataset_path = osp.join('data', 'OGB', 'PPA')
            if exp_mode == 0: # classical
                self.run_experiments = self.run_classical_ogb_experiments
            elif exp_mode == 1: # clustering
                assert prev_res_path is not None
                self.classical_gnn_hyperparameter_experiment_res_path = prev_res_path
                self.run_experiments = self.run_clustering_ogb_experiment
            elif exp_mode == 2: # enhanced model
                assert prev_res_path is not None
                self.clustering_hyperparameter_experiment_res_path = prev_res_path
                self.run_experiments = self.run_ogb_enhanced_gnn_experiment
            elif exp_mode == 3: # full
                self.run_experiments = self.run_ogb_experiments
            self.criterion = torch.nn.CrossEntropyLoss(reduction = 'mean')
        elif dataset_str == 'ogbg-molhiv':
            self.dataset_path = osp.join('data', 'OGB', 'MOL_HIV')
            if exp_mode == 0: # classical
                self.run_experiments = self.run_classical_ogb_experiments
            elif exp_mode == 1: # clustering
                assert prev_res_path is not None
                self.classical_gnn_hyperparameter_experiment_res_path = prev_res_path
                self.run_experiments = self.run_clustering_ogb_experiment
            elif exp_mode == 2: # enhanced model
                assert prev_res_path is not None
                self.clustering_hyperparameter_experiment_res_path = prev_res_path
                self.run_experiments = self.run_ogb_enhanced_gnn_experiment
            elif exp_mode == 3: # full
                self.run_experiments = self.run_ogb_experiments
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif dataset_str in ['NCI1', 'ENZYMES', 'PROTEINS', 'DD', 'CSL']:
            if dataset_str == 'CSL':
                self.dataset_path = osp.join('data', 'CSL', 'CSL_dataset')
            elif dataset_str == 'PROTEINS':
                self.dataset_path = osp.join('data', 'TU', 'PROTEINS')
            elif dataset_str == 'ENZYMES':
                self.dataset_path = osp.join('data', 'TU', 'ENZYMES')
            elif dataset_str == 'NCI1':
                self.dataset_path = osp.join('data', 'TU', 'NCI1')
            elif dataset_str == 'DD':
                self.dataset_path = osp.join('data', 'TU', 'DD')

            if exp_mode == 0: # classical
                self.run_experiments = self.run_classical_csl_prox_experiments
            elif exp_mode == 1: # clustering
                assert prev_res_path is not None
                self.classical_gnn_hyperparameter_experiment_res_path = prev_res_path
                self.run_experiments = self.run_clustering_csl_prox_experiment
            elif exp_mode == 2: # enhanced model
                assert prev_res_path is not None
                self.clustering_hyperparameter_experiment_res_path = prev_res_path
                self.run_experiments = self.run_csl_prox_enhanced_gnn_experiment
            elif exp_mode == 3: # full
                self.run_experiments = self.run_csl_prox_experiments
        elif dataset_str.endswith('-Prox'):
            self.h = int(dataset_str[0])
            assert self.h in [1,3,5,8,10]
            self.dataset_path = osp.join('data', 'Proximity', f'{self.h}-Prox')
            if exp_mode == 0: # classical
                self.run_experiments = self.run_classical_csl_prox_experiments
            elif exp_mode == 1: # clustering
                assert prev_res_path is not None
                self.classical_gnn_hyperparameter_experiment_res_path = prev_res_path
                self.run_experiments = self.run_clustering_csl_prox_experiment
            elif exp_mode == 2: # enhanced model
                assert prev_res_path is not None
                self.clustering_hyperparameter_experiment_res_path = prev_res_path
                self.run_experiments = self.run_csl_prox_enhanced_gnn_experiment
            elif exp_mode == 3: # full
                self.run_experiments = self.run_csl_prox_experiments
        else:
            raise ValueError('Invalid dataset string')
        
        # add description of the examined data
        self.data["dataset"] = self.dataset_str

        if self.is_lovasz_feature:
            datatypestr = "Lovasz"
        else:
            datatypestr = "SP"
        
        self.data["type"] = [datatypestr]

        # Examine the feature type
        if self.k is not None and len(self.k) > 0:
            # use k-disks
            self.data["type"].append("k-Disk")
        elif self.r is not None and self.s is not None and len(r) > 0 and len(s) == len(r):
            self.data["type"].append("r-s-Ring")
        elif self.is_vertex_sp_features:
            self.data["type"].append("vertex_sp")

        print('---   Experiment setup complete   ---')

    def run_classical_ogb_experiments(self):
        t0 = time.time()

        print(f'---   Starting experiments on {self.dataset_str}   ---')

        data = {}
        # data["dataset"] = self.dataset_str
        if self.dataset_str == 'ogbg-molhiv':
            data["loss_func"] = "BCE_with_logits_loss"
            data["metric"] = "rocauc"
        elif self.dataset_str == 'ogbg-ppa':
            data["loss_func"] = "cross_entropy"
            data["metric"] = "acc"
        data["overall_time"] = -1.0
        data["res"] = {}

        print('---   Optimizing classical GNN hyperparameters   ---')
        hyperparameter_opt_data, best_val_perfs, best_test_perfs, best_classic_n_layers, best_classic_n_hidden_channels, best_classic_s_batch, best_classic_n_epoch, best_classic_lr = self.run_ogb_gnn_hyperparameter_optimization(classic_gnn = True)

        val_perfs = torch.tensor(best_val_perfs, dtype = torch.float64)
        test_perfs = torch.tensor(best_test_perfs, dtype = torch.float64)
        val_mean = val_perfs.mean().item()
        if val_perfs.size()[0] == 1:
            val_std = 0.0
        else:
            val_std = val_perfs.std().item()
        test_perfs = torch.tensor(best_test_perfs)
        test_mean = test_perfs.mean().item()
        if test_perfs.size()[0] == 1:
            test_std = 0.0
        else:
            test_std = test_perfs.std().item()

        data["classic_gnn_hyperparameter_opt"] = hyperparameter_opt_data

        data["classic_gnn_hyperparameter_opt"]["res"]["val_perf"]["mean"] = val_mean
        data["classic_gnn_hyperparameter_opt"]["res"]["val_perf"]["std"] = val_std

        data["classic_gnn_hyperparameter_opt"]["res"]["test_perf"]["mean"] = test_mean
        data["classic_gnn_hyperparameter_opt"]["res"]["test_perf"]["std"] = test_std

        data["res"]["hyperparameter"] = {}
        data["res"]["hyperparameter"]["best_num_layers"] = best_classic_n_layers
        data["res"]["hyperparameter"]["best_num_hidden_channels"] = best_classic_n_hidden_channels
        data["res"]["hyperparameter"]["best_batch_size"] = best_classic_s_batch
        data["res"]["hyperparameter"]["best_num_epochs"] = best_classic_n_epoch
        data["res"]["hyperparameter"]["best_lr"] = best_classic_lr

        print(f'---   Val perf: mean: {val_mean}; std: {val_std}   ---')
        print(f'---   Test perf: mean: {test_mean}; std: {test_std}   ---')

        print('---   Optimizing classical GNN hyperparameters complete   ---')

        data["overall_time"] = time.time() - t0

        self.data.update(data)

    def run_classical_csl_prox_experiments(self):
        t0 = time.time()

        print(f'---   Starting experiments on {self.dataset_str}   ---')

        data = {}
        # data["dataset"] = self.dataset_str
        data["loss_func"] = "cross_entropy"
        data["metric"] = "acc"
        data["overall_time"] = -1.0
        data["res"] = {}

        loss_func = F.cross_entropy

        if self.dataset_str in ['NCI1', 'ENZYMES', 'PROTEINS', 'DD']:
            # TU dataset
            dataset = TUDataset(name = self.dataset_str, root = osp.join(self.root_path, self.dataset_path), use_node_attr = False)
            splits = util.generate_tu_splits(dataset = dataset, dataset_path = self.dataset_path, root_path = self.root_path)
        if self.dataset_str == 'CSL':
            dataset = CSL_Dataset(root = osp.join(self.root_path, self.dataset_path))
            splits = dataset.gen_data_splits()
        elif self.dataset_str.endswith('-Prox'):
            dataset = ProximityDataset(root = osp.join(self.root_path, self.dataset_path), h = self.h)
            splits = dataset.gen_data_splits()

        # Optimize GNN hyperparameters without clustering first
        print('---   Optimizing classical GNN hyperparameters   ---')
        hyperparameter_opt_data, best_val_accs, best_test_accs, best_classic_n_layers, best_classic_n_hidden_channels, best_classic_s_batch, best_classic_n_epoch, best_classic_lr = self.run_tu_csl_prox_gnn_hyperparameter_optimization(splits = splits, loss_func = loss_func, classic_gnn = True)

        val_accs = torch.tensor(best_val_accs, dtype = torch.float64)
        test_accs = torch.tensor(best_test_accs, dtype = torch.float64)
        val_mean = val_accs.mean().item()
        if val_accs.size()[0] == 1:
            val_std = 0.0
        else:
            val_std = val_accs.std().item()
        test_mean = test_accs.mean().item()
        if test_accs.size()[0] == 1:
            test_std = 0.0
        else:
            test_std = test_accs.std().item()

        data["classic_gnn_hyperparameter_opt"] = hyperparameter_opt_data

        data["classic_gnn_hyperparameter_opt"]["res"]["val_acc"]["mean"] = val_mean
        data["classic_gnn_hyperparameter_opt"]["res"]["val_acc"]["std"] = val_std

        data["classic_gnn_hyperparameter_opt"]["res"]["test_acc"]["mean"] = test_mean
        data["classic_gnn_hyperparameter_opt"]["res"]["test_acc"]["std"] = test_std

        data["res"]["hyperparameter"] = {}

        data["res"]["hyperparameter"]["best_num_layers"] = best_classic_n_layers
        data["res"]["hyperparameter"]["best_num_hidden_channels"] = best_classic_n_hidden_channels
        data["res"]["hyperparameter"]["best_batch_size"] = best_classic_s_batch
        data["res"]["hyperparameter"]["best_num_epochs"] = best_classic_n_epoch
        data["res"]["hyperparameter"]["best_lr"] = best_classic_lr

        print(f'---   Val acc: mean: {val_mean}; std: {val_std}   ---')
        print(f'---   Test acc: mean: {test_mean}; std: {test_std}   ---')

        print('---   Optimizing classical GNN hyperparameters complete   ---')

        data["overall_time"] = time.time() - t0

        self.data.update(data)

    def run_clustering_ogb_experiment(self) -> None:
        t0 = time.time()

        # Extract the result from the classical GNN experiment
        classical_gnn_hyperparameter_experiment_res_path = self.classical_gnn_hyperparameter_experiment_res_path

        classical_gnn_res_data = util.read_metadata_file(classical_gnn_hyperparameter_experiment_res_path)
        best_classic_n_layers = classical_gnn_res_data["res"]["hyperparameter"]["best_num_layers"]
        best_classic_n_hidden_channels = classical_gnn_res_data["res"]["hyperparameter"]["best_num_hidden_channels"]
        best_classic_s_batch = classical_gnn_res_data["res"]["hyperparameter"]["best_batch_size"]
        best_classic_n_epoch = classical_gnn_res_data["res"]["hyperparameter"]["best_num_epochs"]
        best_classic_lr = classical_gnn_res_data["res"]["hyperparameter"]["best_lr"]

        print(f'---   Starting experiments on {self.dataset_str}   ---')

        data = {}
        # data["dataset"] = self.dataset_str
        if self.dataset_str == 'ogbg-molhiv':
            data["loss_func"] = "BCE_with_logits_loss"
            data["metric"] = "rocauc"
        elif self.dataset_str == 'ogbg-ppa':
            data["loss_func"] = "cross_entropy"
            data["metric"] = "acc"
        data["overall_time"] = -1.0
        data["input_path"] = classical_gnn_hyperparameter_experiment_res_path
        data["res"] = {}

        # Optimize cluster hyperparameter
        clusterer = Vertex_Partition_Clustering(self.root_path)

        print('---   Optimizing clustering hyperparameters   ---')
        # We do not need the explicit best features since we utilise the paths of the best result instead to avoid re-computing the clusterings
        # The max_num_clusters attribute is used since the best clustering might have less than best_num_clusters cluster (due to the min_cluster_size parameter)
        cluster_hyperparameter_opt_data, best_val_perfs, best_test_perfs, best_features_path, best_feature_metadata_filename, best_clustering_path, max_num_clusters, best_num_clusters, best_pca_dim, best_min_cluster_size, best_gpnn_layers, best_gpnn_channels = self.run_ogb_enhanced_gnn_cluster_hyperparameter_optimization(clusterer = clusterer, n_layers = best_classic_n_layers, 
                                                                                                                                                    hidden_channels = best_classic_n_hidden_channels, s_batch = best_classic_s_batch,
                                                                                                                                                    n_epoch = best_classic_n_epoch, lr = best_classic_lr, lo_idx_str = self.lo_idx_str)

        val_perfs = torch.tensor(best_val_perfs, dtype = torch.float64)
        val_mean = val_perfs.mean().item()
        if val_perfs.size()[0] == 1:
            val_std = 0.0
        else:
            val_std = val_perfs.std().item()
        test_perfs = torch.tensor(best_test_perfs, dtype = torch.float64)
        test_mean = test_perfs.mean().item()
        if test_perfs.size()[0] == 1:
            test_std = 0.0
        else:
            test_std = test_perfs.std().item()

        data["clustering_hyperparameter_opt"] = cluster_hyperparameter_opt_data

        data["clustering_hyperparameter_opt"]["res"]["val_perf"]["mean"] = val_mean
        data["clustering_hyperparameter_opt"]["res"]["val_perf"]["std"] = val_std

        data["clustering_hyperparameter_opt"]["res"]["test_perf"]["mean"] = test_mean
        data["clustering_hyperparameter_opt"]["res"]["test_perf"]["std"] = test_std

        data["res"]["hyperparameter"] = {}
        data["res"]["hyperparameter"]["best_num_clusters"] = best_num_clusters
        data["res"]["hyperparameter"]["max_num_clusters"] = max_num_clusters
        data["res"]["hyperparameter"]["best_pca_dim"] = best_pca_dim
        data["res"]["hyperparameter"]["best_min_cluster_size"] = best_min_cluster_size
        data["res"]["hyperparameter"]["best_vertex_feature_metadata_path"] = osp.join(best_features_path, best_feature_metadata_filename)
        data["res"]["hyperparameter"]["best_clustering_metadata_path"] = best_clustering_path
        if self.use_gpnn:
            data["res"]["hyperparameter"]["best_num_gpnn_layers"] = best_gpnn_layers
            data["res"]["hyperparameter"]["best_gpnn_channels"] = best_gpnn_channels

        print(f'---   Val perf: mean: {val_mean}; std: {val_std}   ---')
        print(f'---   Test perf: mean: {test_mean}; std: {test_std}   ---')

        print('---   Optimizing clustering hyperparameters complete   ---')

        data["overall_time"] = time.time() - t0

        self.data.update(data)

    def run_clustering_csl_prox_experiment(self) -> None:
        t0 = time.time()

        # Read previous results
        classical_gnn_hyperparameter_experiment_res_path = self.classical_gnn_hyperparameter_experiment_res_path

        classical_gnn_res_data = util.read_metadata_file(classical_gnn_hyperparameter_experiment_res_path)
        best_classic_n_layers = classical_gnn_res_data["res"]["hyperparameter"]["best_num_layers"]
        best_classic_n_hidden_channels = classical_gnn_res_data["res"]["hyperparameter"]["best_num_hidden_channels"]
        best_classic_s_batch = classical_gnn_res_data["res"]["hyperparameter"]["best_batch_size"]
        best_classic_n_epoch = classical_gnn_res_data["res"]["hyperparameter"]["best_num_epochs"]
        best_classic_lr = classical_gnn_res_data["res"]["hyperparameter"]["best_lr"]

        print(f'---   Starting experiments on {self.dataset_str}   ---')

        data = {}
        # data["dataset"] = self.dataset_str
        data["loss_func"] = "cross_entropy"
        data["metric"] = "acc"
        data["overall_time"] = -1.0
        data["input_path"] = classical_gnn_hyperparameter_experiment_res_path
        data["res"] = {}

        loss_func = F.cross_entropy

        if self.dataset_str in ['NCI1', 'ENZYMES', 'PROTEINS', 'DD']:
            # TU dataset
            dataset = TUDataset(name = self.dataset_str, root = osp.join(self.root_path, self.dataset_path), use_node_attr = False)
            splits = util.generate_tu_splits(dataset = dataset, dataset_path = self.dataset_path, root_path = self.root_path)
        elif self.dataset_str == 'CSL':
            dataset = CSL_Dataset(root = osp.join(self.root_path, self.dataset_path))
            splits = dataset.gen_data_splits()
        elif self.dataset_str.endswith('-Prox'):
            dataset = ProximityDataset(root = osp.join(self.root_path, self.dataset_path), h = self.h)
            splits = dataset.gen_data_splits()


        # Optimize cluster hyperparameter
        clusterer = Vertex_Partition_Clustering(self.root_path)

        print('---   Optimizing clustering hyperparameters   ---')
        # We do not need the explicit best features since we utilise the paths of the best result instead to avoid re-computing the clusterings
        # The max_num_clusters attribute is used since the best clustering might have less than best_num_clusters cluster (due to the min_cluster_size parameter)
        cluster_hyperparameter_opt_data, best_val_accs, best_test_accs, best_features_path, best_feature_metadata_filename, best_clustering_paths, max_num_clusters, best_num_clusters, best_pca_dim, best_min_cluster_size, best_gpnn_layers, best_gpnn_channels = self.run_csl_prox_enhanced_gnn_cluster_hyperparameter_optimization(clusterer = clusterer, n_layers = best_classic_n_layers, 
                                                                                                                                                       hidden_channels = best_classic_n_hidden_channels, s_batch = best_classic_s_batch,
                                                                                                                                                       n_epoch = best_classic_n_epoch, lr = best_classic_lr, splits = splits, loss_func = loss_func, lo_idx_str = self.lo_idx_str)

        val_accs = torch.tensor(best_val_accs, dtype = torch.float64)
        test_accs = torch.tensor(best_test_accs, dtype = torch.float64)
        val_mean = val_accs.mean().item()
        if val_accs.size()[0] == 1:
            val_std = 0.0
        else:
            val_std = val_accs.std().item()
        test_mean = test_accs.mean().item()
        if test_accs.size()[0] == 1:
            test_std = 0.0
        else:
            test_std = test_accs.std().item()

        data["clustering_hyperparameter_opt"] = cluster_hyperparameter_opt_data

        data["clustering_hyperparameter_opt"]["res"]["val_acc"]["mean"] = val_mean
        data["clustering_hyperparameter_opt"]["res"]["val_acc"]["std"] = val_std

        data["clustering_hyperparameter_opt"]["res"]["test_acc"]["mean"] = test_mean
        data["clustering_hyperparameter_opt"]["res"]["test_acc"]["std"] = test_std

        print(f'---   Val acc: mean: {val_mean}; std: {val_std}   ---')
        print(f'---   Test acc: mean: {test_mean}; std: {test_std}   ---')

        data["res"]["hyperparameter"] = {}
        data["res"]["hyperparameter"]["best_num_clusters"] = best_num_clusters
        data["res"]["hyperparameter"]["max_num_clusters"] = max_num_clusters
        data["res"]["hyperparameter"]["best_pca_dim"] = best_pca_dim
        data["res"]["hyperparameter"]["best_min_cluster_size"] = best_min_cluster_size
        data["res"]["hyperparameter"]["best_vertex_feature_metadata_path"] = osp.join(best_features_path, best_feature_metadata_filename)
        data["res"]["hyperparameter"]["best_clustering_metadata_paths"] = best_clustering_paths
        if self.use_gpnn:
            data["res"]["hyperparameter"]["best_num_gpnn_layers"] = best_gpnn_layers
            data["res"]["hyperparameter"]["best_gpnn_channels"] = best_gpnn_channels

        print('---   Optimizing clustering hyperparameters complete   ---')

        data["overall_time"] = time.time() - t0

        self.data.update(data)

    def run_ogb_enhanced_gnn_experiment(self) -> None:
        t0 = time.time()

        # Read results from previous step
        clustering_hyperparameter_experiment_res_path = self.clustering_hyperparameter_experiment_res_path

        clustering_experiment_data = util.read_metadata_file(clustering_hyperparameter_experiment_res_path)
        max_num_clusters = clustering_experiment_data["res"]["hyperparameter"]["max_num_clusters"]
        vertex_feature_metadata_path = clustering_experiment_data["res"]["hyperparameter"]["best_vertex_feature_metadata_path"]
        best_clustering_metadata_path = clustering_experiment_data["res"]["hyperparameter"]["best_clustering_metadata_path"]
        if self.use_gpnn:
            best_gpnn_layers = clustering_experiment_data["res"]["hyperparameter"]["best_num_gpnn_layers"]
            best_gpnn_channels = clustering_experiment_data["res"]["hyperparameter"]["best_gpnn_channels"]
        else:
            best_gpnn_layers = None
            best_gpnn_channels = None

        print(f'---   Starting experiments on {self.dataset_str}   ---')

        data = {}
        # data["dataset"] = self.dataset_str
        if self.dataset_str == 'ogbg-molhiv':
            data["loss_func"] = "BCE_with_logits_loss"
            data["metric"] = "rocauc"
        elif self.dataset_str == 'ogbg-ppa':
            data["loss_func"] = "cross_entropy"
            data["metric"] = "acc"
        data["overall_time"] = -1.0
        data["input_path"] = clustering_hyperparameter_experiment_res_path
        data["res"] = {}

        print('---   Running hyperparameter optimization for enhanced GNN   ---')
        # Re-train enhanced gnn hyperparameter using clustering hyperparameter
        enhanced_hyperparameter_opt_data, best_val_perfs, best_test_perfs, best_n_layers, best_n_hidden_channels, best_s_batch, best_n_epoch, best_lr = self.run_ogb_gnn_hyperparameter_optimization(classic_gnn = False, best_clustering_path = best_clustering_metadata_path, best_num_clusters = max_num_clusters, vertex_feature_metadata_path = vertex_feature_metadata_path, best_num_gpnn_layers = best_gpnn_layers, best_gpnn_channels = best_gpnn_channels)

        data["enhanced_gnn_hyperparameter_opt"] = enhanced_hyperparameter_opt_data

        val_perfs = torch.tensor(best_val_perfs, dtype = torch.float64)
        # test_perfs = torch.tensor(best_test_perfs, dtype = torch.float64)
        val_mean = val_perfs.mean().item()
        if val_perfs.size()[0] == 1:
            val_std = 0.0
        else:
            val_std = val_perfs.std().item()
        test_perfs = torch.tensor(best_test_perfs, dtype = torch.float64)
        test_mean = test_perfs.mean().item()
        if test_perfs.size()[0] == 1:
            test_std = 0.0
        else:
            test_std = test_perfs.std().item()

        data["enhanced_gnn_hyperparameter_opt"]["res"]["val_perf"]["mean"] = val_mean
        data["enhanced_gnn_hyperparameter_opt"]["res"]["val_perf"]["std"] = val_std

        data["enhanced_gnn_hyperparameter_opt"]["res"]["test_perf"]["mean"] = test_mean
        data["enhanced_gnn_hyperparameter_opt"]["res"]["test_perf"]["std"] = test_std

        data["res"]["test_perf"] = {}
        data["res"]["test_perf"]["mean"] = test_mean
        data["res"]["test_perf"]["std"] = test_std
        data["res"]["val_perf"] = {}
        data["res"]["val_perf"]["mean"] = val_mean
        data["res"]["val_perf"]["std"] = val_std
        
        data["res"]["hyperparameter"] = {}

        data["res"]["hyperparameter"]["best_num_layers"] = best_n_layers
        data["res"]["hyperparameter"]["best_num_hidden_channels"] = best_n_hidden_channels
        data["res"]["hyperparameter"]["best_batch_size"] = best_s_batch
        data["res"]["hyperparameter"]["best_num_epochs"] = best_n_epoch
        data["res"]["hyperparameter"]["best_lr"] = best_lr

        data["res"]["hyperparameter"]["best_num_clusters"] = clustering_experiment_data["res"]["hyperparameter"]["best_num_clusters"]
        data["res"]["hyperparameter"]["max_num_clusters"] = max_num_clusters
        data["res"]["hyperparameter"]["best_pca_dim"] = clustering_experiment_data["res"]["hyperparameter"]["best_pca_dim"]
        data["res"]["hyperparameter"]["best_min_cluster_size"] = clustering_experiment_data["res"]["hyperparameter"]["best_min_cluster_size"]
        if self.use_gpnn:
            data["res"]["hyperparameter"]["best_num_gpnn_layers"] = clustering_experiment_data["res"]["hyperparameter"]["best_num_gpnn_layers"]
            data["res"]["hyperparameter"]["best_num_gpnn_layers"] = clustering_experiment_data["res"]["hyperparameter"]["best_gpnn_channels"]

        print(f'---   Val perf: mean: {val_mean}; std: {val_std}   ---')
        print(f'---   Test perf: mean: {test_mean}; std: {test_std}   ---')

        print('---   Running hyperparameter optimization for enhanced GNN completed   ---')

        data["overall_time"] = time.time() - t0

    def run_csl_prox_enhanced_gnn_experiment(self) -> None:
        t0 = time.time()

        # Read results from previous step
        clustering_hyperparameter_experiment_res_path = self.clustering_hyperparameter_experiment_res_path

        clustering_experiment_data = util.read_metadata_file(clustering_hyperparameter_experiment_res_path)
        max_num_clusters = clustering_experiment_data["res"]["hyperparameter"]["max_num_clusters"]
        vertex_feature_metadata_path = clustering_experiment_data["res"]["hyperparameter"]["best_vertex_feature_metadata_path"]
        best_clustering_metadata_paths = clustering_experiment_data["res"]["hyperparameter"]["best_clustering_metadata_paths"]
        if self.use_gpnn:
            best_gpnn_layers = clustering_experiment_data["res"]["hyperparameter"]["best_num_gpnn_layers"]
            best_gpnn_channels = clustering_experiment_data["res"]["hyperparameter"]["best_gpnn_channels"]
        else:
            best_gpnn_layers = None
            best_gpnn_channels = None

        print(f'---   Starting experiments on {self.dataset_str}   ---')

        data = {}
        # data["dataset"] = self.dataset_str
        data["loss_func"] = "cross_entropy"
        data["metric"] = "acc"
        data["overall_time"] = -1.0
        data["input_path"] = clustering_hyperparameter_experiment_res_path
        data["res"] = {}

        loss_func = F.cross_entropy

        if self.dataset_str in ['NCI1', 'ENZYMES', 'PROTEINS', 'DD']:
            # TU dataset
            dataset = TUDataset(name = self.dataset_str, root = osp.join(self.root_path, self.dataset_path), use_node_attr = False)
            splits = util.generate_tu_splits(dataset = dataset, dataset_path = self.dataset_path, root_path = self.root_path)
        elif self.dataset_str == 'CSL':
            dataset = CSL_Dataset(root = osp.join(self.root_path, self.dataset_path))
            splits = dataset.gen_data_splits()
        elif self.dataset_str.endswith('-Prox'):
            dataset = ProximityDataset(root = osp.join(self.root_path, self.dataset_path), h = self.h)
            splits = dataset.gen_data_splits()

        print('---   Running hyperparameter optimization for enhanced GNN   ---')
        # Re-train enhanced gnn hyperparameter using clustering hyperparameter
        enhanced_hyperparameter_opt_data, best_val_accs, best_test_accs, best_n_layers, best_n_hidden_channels, best_s_batch, best_n_epoch, best_lr = self.run_tu_csl_prox_gnn_hyperparameter_optimization(splits = splits, loss_func = loss_func, classic_gnn = False, best_clustering_paths = best_clustering_metadata_paths, best_num_clusters = max_num_clusters, best_vertex_feature_metadata_path = vertex_feature_metadata_path, best_num_gpnn_layers = best_gpnn_layers, best_gpnn_channels = best_gpnn_channels)

        data["enhanced_gnn_hyperparameter_opt"] = enhanced_hyperparameter_opt_data

        val_accs = torch.tensor(best_val_accs, dtype = torch.float64)
        test_accs = torch.tensor(best_test_accs, dtype = torch.float64)
        val_mean = val_accs.mean().item()
        if val_accs.size()[0] == 1:
            val_std = 0.0
        else:
            val_std = val_accs.std().item()
        test_mean = test_accs.mean().item()
        if test_accs.size()[0] == 1:
            test_std = 0.0
        else:
            test_std = test_accs.std().item()

        data["enhanced_gnn_hyperparameter_opt"]["res"]["val_acc"]["mean"] = val_mean
        data["enhanced_gnn_hyperparameter_opt"]["res"]["val_acc"]["std"] = val_std

        data["enhanced_gnn_hyperparameter_opt"]["res"]["test_acc"]["mean"] = test_mean
        data["enhanced_gnn_hyperparameter_opt"]["res"]["test_acc"]["std"] = test_std

        data["res"]["test_acc"] = {}
        data["res"]["test_acc"]["mean"] = test_mean
        data["res"]["test_acc"]["std"] = test_std
        data["res"]["val_acc"] = {}
        data["res"]["val_acc"]["mean"] = val_mean
        data["res"]["val_acc"]["std"] = val_std
        
        data["res"]["hyperparameter"] = {}

        data["res"]["hyperparameter"]["best_num_layers"] = best_n_layers
        data["res"]["hyperparameter"]["best_num_hidden_channels"] = best_n_hidden_channels
        data["res"]["hyperparameter"]["best_batch_size"] = best_s_batch
        data["res"]["hyperparameter"]["best_num_epochs"] = best_n_epoch
        data["res"]["hyperparameter"]["best_lr"] = best_lr

        data["res"]["hyperparameter"]["best_num_clusters"] = clustering_experiment_data["res"]["hyperparameter"]["best_num_clusters"]
        data["res"]["hyperparameter"]["max_num_clusters"] = max_num_clusters
        data["res"]["hyperparameter"]["best_pca_dim"] = clustering_experiment_data["res"]["hyperparameter"]["best_pca_dim"]
        data["res"]["hyperparameter"]["best_min_cluster_size"] = clustering_experiment_data["res"]["hyperparameter"]["best_min_cluster_size"]
        if self.use_gpnn:
            data["res"]["hyperparameter"]["best_num_gpnn_layers"] = clustering_experiment_data["res"]["hyperparameter"]["best_num_gpnn_layers"]
            data["res"]["hyperparameter"]["best_gpnn_channels"] = clustering_experiment_data["res"]["hyperparameter"]["best_gpnn_channels"]

        print(f'---   Val acc: mean: {val_mean}; std: {val_std}   ---')
        print(f'---   Test acc: mean: {test_mean}; std: {test_std}   ---')

        print('---   Running hyperparameter optimization for enhanced GNN completed   ---')

        data["overall_time"] = time.time() - t0

        self.data.update(data)

    # Schedules multiple experiments
    def run_ogb_experiments(self) -> None:

        t0 = time.time()

        print(f'---   Starting experiments on {self.dataset_str}   ---')

        data = {}
        # data["dataset"] = self.dataset_str
        if self.dataset_str == 'ogbg-molhiv':
            data["loss_func"] = "BCE_with_logits_loss"
            data["metric"] = "rocauc"
        elif self.dataset_str == 'ogbg-ppa':
            data["loss_func"] = "cross_entropy"
            data["metric"] = "acc"
        data["overall_time"] = -1.0
        data["res"] = {}

        # if self.dataset_str == 'ogbg-molhiv':
        #     dataset = PygGraphPropPredDataset(name = 'ogbg-molhiv', root = osp.join(self.root_path, self.dataset_path))
        # elif self.dataset_str.endswith('-Prox'):
        #     dataset = ProximityDataset(root = osp.join(self.root_path, self.dataset_path), h = self.h)

        # splits = dataset.gen_data_splits()

        # Optimize GNN hyperparameters without clustering first
        print('---   Optimizing classical GNN hyperparameters   ---')
        hyperparameter_opt_data, best_val_perfs, best_test_perfs, best_classic_n_layers, best_classic_n_hidden_channels, best_classic_s_batch, best_classic_n_epoch, best_classic_lr = self.run_ogb_gnn_hyperparameter_optimization(classic_gnn = True)

        val_perfs = torch.tensor(best_val_perfs, dtype = torch.float64)
        # test_perfs = torch.tensor(best_test_perfs)
        val_mean = val_perfs.mean().item()
        if val_perfs.size()[0] == 1:
            val_std = 0.0
        else:
            val_std = val_perfs.std().item()
        test_perfs = torch.tensor(best_test_perfs, dtype = torch.float64)
        test_mean = test_perfs.mean().item()
        if test_perfs.size()[0] == 1:
            test_std = 0.0
        else:
            test_std = test_perfs.std().item()

        data["classic_gnn_hyperparameter_opt"] = hyperparameter_opt_data

        data["classic_gnn_hyperparameter_opt"]["res"]["val_perf"]["mean"] = val_mean
        data["classic_gnn_hyperparameter_opt"]["res"]["val_perf"]["std"] = val_std

        data["classic_gnn_hyperparameter_opt"]["res"]["test_perf"]["mean"] = test_mean
        data["classic_gnn_hyperparameter_opt"]["res"]["test_perf"]["std"] = test_std

        print(f'---   Val perf: mean: {val_mean}; std: {val_std}   ---')
        print(f'---   Test perf: mean: {test_mean}; std: {test_std}   ---')

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
        cluster_hyperparameter_opt_data, best_val_perfs, best_test_perfs, best_features_path, best_feature_metadata_filename, best_clustering_path, max_num_clusters, best_num_clusters, best_pca_dim, best_min_cluster_size, best_gpnn_layers, best_gpnn_channels = self.run_ogb_enhanced_gnn_cluster_hyperparameter_optimization(clusterer = clusterer, n_layers = best_classic_n_layers, 
                                                                                                                                                    hidden_channels = best_classic_n_hidden_channels, s_batch = best_classic_s_batch,
                                                                                                                                                    n_epoch = best_classic_n_epoch, lr = best_classic_lr, lo_idx_str = self.lo_idx_str)

        val_perfs = torch.tensor(best_val_perfs, dtype = torch.float64)
        val_mean = val_perfs.mean().item()
        if val_perfs.size()[0] == 1:
            val_std = 0.0
        else:
            val_std = val_perfs.std().item()
        test_perfs = torch.tensor(best_test_perfs, dtype = torch.float64)
        test_mean = test_perfs.mean().item()
        if test_perfs.size()[0] == 1:
            test_std = 0.0
        else:
            test_std = test_perfs.std().item()

        data["clustering_hyperparameter_opt"] = cluster_hyperparameter_opt_data

        data["clustering_hyperparameter_opt"]["res"]["val_perf"]["mean"] = val_mean
        data["clustering_hyperparameter_opt"]["res"]["val_perf"]["std"] = val_std

        data["clustering_hyperparameter_opt"]["res"]["test_perf"]["mean"] = test_mean
        data["clustering_hyperparameter_opt"]["res"]["test_perf"]["std"] = test_std

        print(f'---   Val perf: mean: {val_mean}; std: {val_std}   ---')
        print(f'---   Test perf: mean: {test_mean}; std: {test_std}   ---')

        print('---   Optimizing clustering hyperparameters complete   ---')

        # # Testing
        # best_features_path = osp.join(self.dataset_path, 'results', 'vertex_SP_features')
        # best_clustering_paths = {}
        # for idx in range(5):
        #     best_clustering_paths [idx] = osp.join(best_features_path, 'cluster-gnn', f'{idx}-fold', '4-means_min6140-size_cluster_metadata.json')
        # max_num_clusters = 1

        print('---   Running hyperparameter optimization for final enhanced GNN   ---')
        # Re-train enhanced gnn hyperparameter using clustering hyperparameter
        enhanced_hyperparameter_opt_data, best_val_perfs, best_test_perfs, best_n_layers, best_n_hidden_channels, best_s_batch, best_n_epoch, best_lr = self.run_ogb_gnn_hyperparameter_optimization(classic_gnn = False, best_clustering_path = best_clustering_path, best_num_clusters = max_num_clusters, vertex_feature_metadata_path = osp.join(best_features_path, best_feature_metadata_filename), best_gpnn_channels = best_gpnn_channels, best_num_gpnn_layers = best_gpnn_layers)

        data["enhanced_gnn_hyperparameter_opt"] = enhanced_hyperparameter_opt_data

        val_perfs = torch.tensor(best_val_perfs, dtype = torch.float64)
        # test_perfs = torch.tensor(best_test_perfs)
        val_mean = val_perfs.mean().item()
        if val_perfs.size()[0] == 1:
            val_std = 0.0
        else:
            val_std = val_perfs.std().item()
        test_perfs = torch.tensor(best_test_perfs, dtype = torch.float64)
        test_mean = test_perfs.mean().item()
        if test_perfs.size()[0] == 1:
            test_std = 0.0
        else:
            test_std = test_perfs.std().item()

        data["enhanced_gnn_hyperparameter_opt"]["res"]["val_perf"]["mean"] = val_mean
        data["enhanced_gnn_hyperparameter_opt"]["res"]["val_perf"]["std"] = val_std

        data["enhanced_gnn_hyperparameter_opt"]["res"]["test_perf"]["mean"] = test_mean
        data["enhanced_gnn_hyperparameter_opt"]["res"]["test_perf"]["std"] = test_std

        data["res"]["test_perf"] = {}
        data["res"]["test_perf"]["mean"] = test_mean
        data["res"]["test_perf"]["std"] = test_std
        data["res"]["val_perf"] = {}
        data["res"]["val_perf"]["mean"] = val_mean
        data["res"]["val_perf"]["std"] = val_std
        
        data["res"]["hyperparameter"] = {}

        data["res"]["hyperparameter"]["best_num_layers"] = best_n_layers
        data["res"]["hyperparameter"]["best_num_hidden_channels"] = best_n_hidden_channels
        data["res"]["hyperparameter"]["best_batch_size"] = best_s_batch
        data["res"]["hyperparameter"]["best_num_epochs"] = best_n_epoch
        data["res"]["hyperparameter"]["best_lr"] = best_lr

        data["res"]["hyperparameter"]["best_num_clusters"] = best_num_clusters
        data["res"]["hyperparameter"]["max_num_clusters"] = max_num_clusters
        data["res"]["hyperparameter"]["best_pca_dim"] = best_pca_dim
        data["res"]["hyperparameter"]["best_min_cluster_size"] = best_min_cluster_size
        if self.use_gpnn:
            data["res"]["hyperparameter"]["best_num_gpnn_layers"] = best_gpnn_layers
            data["res"]["hyperparameter"]["best_gpnn_channels"] = best_gpnn_channels

        print(f'---   Val perf: mean: {val_mean}; std: {val_std}   ---')
        print(f'---   Test perf: mean: {test_mean}; std: {test_std}   ---')

        print('---   Running hyperparameter optimization for final enhanced GNN completed   ---')

        data["overall_time"] = time.time() - t0

        self.data.update(data)

    def run_ogb_gnn_hyperparameter_optimization(self, classic_gnn: bool = True, best_clustering_path: Optional[str] = None, best_num_clusters: Optional[int] = None, vertex_feature_metadata_path: Optional[str] = None, best_num_gpnn_layers: Optional[int] = None, best_gpnn_channels: Optional[int] = None) -> Tuple[Dict, GNN_Manager, int, int, int, int, float]:
        data = {}

        num_experiments = (len(self.num_layers) * len(self.hidden_channels) * len(self.batch_sizes) * len(self.num_epochs) * len(self.lrs))
        data["num_experiments"] = num_experiments

        # # Will be overwritten if training enhanced gnn hyperparameters
        # if self.dataset_str == 'CSL':
        #     dataset = CSL_Dataset(root = osp.join(self.root_path, self.dataset_path))
        # elif self.dataset_str.endswith('-Prox'):
        #     dataset = ProximityDataset(root = osp.join(self.root_path, self.dataset_path), h = self.h)

        # The best hyperparameters
        data["res"] = {}
        data["res"]["best_avg_val_perf"] = -1.0
        data["res"]["best_experiment_idx"] = -1
        data["res"]["best_num_layers"] = -1
        data["res"]["best_hidden_channels"] = -1
        data["res"]["best_batch_size"] = -1
        data["res"]["best_num_epochs"] = -1
        data["res"]["best_lr"] = -1.0
        # Set in the parent method
        data["res"]["val_perf"] = {}
        data["res"]["val_perf"]["mean"] = -1.0
        data["res"]["val_perf"]["std"] = -1.0

        data["res"]["test_perf"] = {}
        data["res"]["test_perf"]["mean"] = -1.0
        data["res"]["test_perf"]["std"] = -1.0

        # Stores the data for a single run
        data["experiment_idx"] = {}

        # Stores the average time used for computing a split, a rerun and an epoch (averaged over all total runs (including all folds))
        data["times"] = {}
        data["times"]["experiment_avg"] = -1.0
        data["times"]["rerun_avg"] = -1.0
        data["times"]["epoch_avg"] = -1.0

        # iterate over the hyperparameters for this step
        best_avg_val_perf = 0.0
        best_n_layers = 0
        best_n_hidden_channels = 0
        best_s_batch = 0
        best_n_epoch = 0
        best_lr = -1.0
        best_val_perf_experiment_idx = 0

        best_val_perfs = []
        best_test_perfs = []

        avg_experiment_time = 0.0
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
                            data["experiment_idx"][cur_experiment_idx]["avg_val_perf"] = -1.0

                            # Load dataset and split
                            dataset = PygGraphPropPredDataset(name = self.dataset_str, root = osp.join(self.root_path, self.dataset_path))

                            split_idx = dataset.get_idx_split()
                            train_indices = split_idx["train"]
                            val_indices = split_idx["valid"]
                            test_indices = split_idx["test"]

                            split_prop = { "desc" : f"fixed_{self.dataset_str}", "split_mode" : "fixed", "num_samples" : { "train" : train_indices.size()[0], "val" : val_indices.size()[0], "test" : test_indices.size()[0] }}

                            evaluator = Evaluator(self.dataset_str)

                            if classic_gnn:
                                self.gnn.set_dataset_parameters(num_classes = dataset.num_tasks, num_features = dataset.num_features, num_clusters = 0)
                                self.load_classic_gnn(hidden_channels = n_hidden_channels, num_layers = n_layers, lr = lr)
                                desc_str = f'{self.dataset_str}_classic_{self.model_str}_exp-{cur_experiment_idx}'
                            else:
                                self.gnn.set_dataset_parameters(num_classes = dataset.num_tasks, num_features = dataset.num_features, num_clusters = best_num_clusters)
                                if self.use_gpnn:
                                    self.load_enhanced_gnn(num_gpnn_layers = best_num_gpnn_layers, gpnn_channels = best_gpnn_channels, gnn_hidden_channels = n_hidden_channels, base_gnn_str = self.base_model, num_gnn_layers = n_layers, lr = lr)
                                else:
                                    self.load_enhanced_gnn(hidden_channels = n_hidden_channels, num_layers = n_layers, lr = lr)
                                desc_str = f'{self.dataset_str}_enhanced_{self.model_str}_exp-{cur_experiment_idx}/{num_experiments}'

                            data["experiment_idx"][cur_experiment_idx]["model"] = self.gnn.get_metadata()

                            # data_folds stores the data for a single fold
                            data_folds = {}
                            data_folds["split_prop"] = split_prop

                            # Enhance the dataset with clustering_ids (using the previously computed information regarding the folds)
                            if not classic_gnn:
                                feature_metadata_path = osp.join(vertex_feature_metadata_path)
                                dataset, _ = gnn_utils.include_cluster_id_feature_transform(dataset = dataset, absolute_path_prefix = self.root_path, vertex_feature_metadata_path = feature_metadata_path, cluster_metadata_path = best_clustering_path)
                                # dataset, _ = gnn_utils.include_cluster_id_feature_transform(dataset = dataset, absolute_path_prefix = self.root_path, feature_metadata_path = feature_metadata_path, cluster_metadata_path = best_clustering_path)

                            rerun_data = {}

                            avg_rerun_time = 0.0
                            avg_rerun_epoch_time = 0.0

                            test_perfs = []
                            val_perfs = []

                            for cur_rerun in range(self.num_reruns):

                                start_rerun = time.time()

                                rerun_data[cur_rerun] = {}
                                rerun_data[cur_rerun]["best_val_perf"] = -1.0
                                rerun_data[cur_rerun]["best_val_perf_epoch_idx"] = -1
                                rerun_data[cur_rerun]["epoch"] = {}

                                self.gnn.model.reset_parameters()

                                # We shuffle the training data for training
                                train_loader = DataLoader(dataset = dataset[train_indices], batch_size = s_batch, shuffle = True, num_workers = constants.num_workers)
                                val_loader = DataLoader(dataset = dataset[val_indices], batch_size = s_batch, shuffle = False, num_workers = constants.num_workers)
                                test_loader = DataLoader(dataset = dataset[test_indices], batch_size = s_batch, shuffle = False, num_workers = constants.num_workers)

                                best_rerun_val_perf = -1.0
                                best_rerun_test_perf = -1.0
                                best_rerun_val_epoch = 0

                                avg_epoch_time = 0.0

                                epoch_patience = 0
                                stop = False
                                total_num_epoch = 0

                                for epoch in range(n_epoch):
                                    if stop:
                                        break

                                    t0 = time.time()

                                    rerun_data[cur_rerun]["epoch"][epoch] = {}

                                    # train the gnn
                                    self.train_ogb(gnn = self.gnn, loader = train_loader, desc = f'{desc_str}_rerun-{cur_rerun}/{self.num_reruns}_epoch-{epoch}/{n_epoch}_train')

                                    # evaluate the results
                                    train_perf = self.eval_ogb(gnn = self.gnn, loader = train_loader, evaluator = evaluator, desc = f'{desc_str}_rerun-{cur_rerun}/{self.num_reruns}_epoch-{epoch}/{n_epoch}_train_eval')[dataset.eval_metric].item()
                                    val_perf = self.eval_ogb(gnn = self.gnn, loader = val_loader, evaluator = evaluator, desc = f'{desc_str}_rerun-{cur_rerun}/{self.num_reruns}_epoch-{epoch}/{n_epoch}_val_eval')[dataset.eval_metric].item()
                                    test_perf = self.eval_ogb(gnn = self.gnn, loader = test_loader, evaluator = evaluator, desc = f'{desc_str}_rerun-{cur_rerun}/{self.num_reruns}_epoch-{epoch}/{n_epoch}_test_eval')[dataset.eval_metric].item()

                                    rerun_data[cur_rerun]["epoch"][epoch]["perf"] = {'Train': train_perf, 'Validation': val_perf, 'Test': test_perf}

                                    if best_rerun_val_perf < val_perf:
                                        best_rerun_val_perf = val_perf
                                        best_rerun_val_epoch = epoch
                                        best_rerun_test_perf = test_perf
                                        epoch_patience = 0
                                    else:
                                        epoch_patience += 1
                                        if epoch_patience >= self.max_patience:
                                            stop = True
                                    
                                    total_num_epoch += 1

                                    time_epoch = time.time() - t0
                                    rerun_data[cur_rerun]["epoch"][epoch]["time"] = time_epoch
                                    avg_epoch_time += time_epoch
                                
                                avg_epoch_time /= total_num_epoch
                                avg_rerun_epoch_time += avg_epoch_time

                                rerun_data[cur_rerun]["best_val_perf"] = best_rerun_val_perf
                                rerun_data[cur_rerun]["best_val_perf_epoch_idx"] = best_rerun_val_epoch

                                val_perfs.append(best_rerun_val_perf)
                                test_perfs.append(best_rerun_test_perf)
                                avg_rerun_time += time.time() - start_rerun

                            avg_rerun_time /= self.num_reruns
                            avg_rerun_epoch_time /= self.num_reruns

                            mean_val_perf = torch.tensor(val_perfs).mean().item()

                            # store data
                            data_folds["rerun"] = rerun_data
                            
                            data["experiment_idx"][cur_experiment_idx]["avg_val_perf"] = mean_val_perf
                            data["experiment_idx"][cur_experiment_idx].update(data_folds)

                            # These hyperparameters yield the best results so far
                            if best_avg_val_perf < mean_val_perf:
                                best_avg_val_perf = mean_val_perf
                                best_n_layers = n_layers
                                best_n_hidden_channels = n_hidden_channels
                                best_s_batch = s_batch
                                best_n_epoch = n_epoch
                                best_lr = lr
                                best_val_perf_experiment_idx = cur_experiment_idx
                                best_val_perfs = val_perfs
                                best_test_perfs = test_perfs
                            
                            cur_experiment_idx += 1

                            avg_rerun_time_overall += avg_rerun_time
                            avg_epoch_time_overall += avg_rerun_epoch_time
                            avg_experiment_time += time.time() - experiment_start

        # Store results
        data["res"]["best_avg_val_perf"] = best_avg_val_perf
        data["res"]["best_experiment_idx"] = best_val_perf_experiment_idx
        data["res"]["best_num_layers"] = best_n_layers
        data["res"]["best_hidden_channels"] = best_n_hidden_channels
        data["res"]["best_batch_size"] = best_s_batch
        data["res"]["best_num_epochs"] = best_n_epoch
        data["res"]["best_lr"] = best_lr

        avg_experiment_time /= num_experiments
        avg_rerun_time_overall /= num_experiments
        avg_epoch_time_overall /= num_experiments

        data["times"]["experiment_avg"] = avg_experiment_time
        data["times"]["rerun_avg"] = avg_rerun_time_overall
        data["times"]["epoch_avg"] = avg_epoch_time_overall

        return data, best_val_perfs, best_test_perfs, best_n_layers, best_n_hidden_channels, best_s_batch, best_n_epoch, best_lr


    def run_ogb_enhanced_gnn_cluster_hyperparameter_optimization(self, clusterer: Vertex_Partition_Clustering, n_layers: int, hidden_channels: int, s_batch: int, n_epoch: int, lr: float, lo_idx_str: Optional[str] = "") -> Tuple[Dict, str, Dict, int, int, int]:
        data = {}
        num_experiments = -1

        # A list of paths to all the vertex feature directories that should be considered for hyperparameter optimization
        vertex_feature_paths = []
        metadata_filenames = []

        # decide whether k-disks, r-s-rings or vertex_sp_features are used.
        if self.k is not None and len(self.k) > 0:
            # Use k-disks
            num_experiments = len(self.num_clusters) * len(self.pca_dims) * len(self.min_cluster_sizes) * len(self.k)
            if self.use_gpnn:
                num_experiments *= len(self.gpnn_channels) * len(self.gpnn_layers)
            for k in self.k:
                if self.is_lovasz_feature:
                    vertex_feature_paths.append(osp.join(self.dataset_path, 'results', f'{k}-disk_lo_features'))
                    if self.dataset_str == "ogbg-molhiv":
                        metadata_filenames.append(f"MOLHIV_{k}-disk_lo_features{lo_idx_str}_metadata.json")
                    elif self.dataset_str == "ogbg-ppa":
                        metadata_filenames.append(f"PPA_{k}-disk_lo_features{lo_idx_str}_metadata.json")
                    # elif self.dataset_str == "CSL":
                    #     metadata_filenames.append(f"CSL_{k}-disk_lo_features{lo_idx_str}_metadata.json")
                    # elif self.dataset_str.endswith("-Prox"):
                    #     metadata_filenames.append(f"{self.h}-Prox_{k}-disk_lo_features{lo_idx_str}_metadata.json")
                else:
                    vertex_feature_paths.append(osp.join(self.dataset_path, 'results', f'{k}-disk_SP_features'))
                    if self.dataset_str == "ogbg-molhiv":
                        metadata_filenames.append(f"MOLHIV_{k}-disk_SP_features_metadata.json")
                    elif self.dataset_str == "ogbg-ppa":
                        metadata_filenames.append(f"PPA_{k}-disk_SP_features_metadata.json")
                    # elif self.dataset_str == "CSL":
                    #     metadata_filenames.append(f"CSL_{k}-disk_SP_features_metadata.json")
                    # elif self.dataset_str.endswith("-Prox"):
                    #     metadata_filenames.append(f"{self.h}-Prox_{k}-disk_SP_features_metadata.json")

        elif self.r is not None and self.s is not None and len(self.r) > 0 and len(self.s) == len(self.r):
            # Use r-s-rings
            num_experiments = len(self.num_clusters) * len(self.pca_dims) * len(self.min_cluster_sizes) * len(self.r)
            if self.use_gpnn:
                num_experiments *= len(self.gpnn_channels) * len(self.gpnn_layers)
            for idx in range(len(self.r)):
                if self.is_lovasz_feature:
                    vertex_feature_paths.append(osp.join(self.dataset_path, 'results', f'{self.r[idx]}-{self.s[idx]}-ring_lo_features'))
                    if self.dataset_str == "ogbg-molhiv":
                        metadata_filenames.append(f"MOLHIV_{self.r[idx]}-{self.s[idx]}-ring_lo_features{lo_idx_str}_metadata.json")
                    elif self.dataset_str == "ogbg-ppa":
                        metadata_filenames.append(f"PPA_{self.r[idx]}-{self.s[idx]}-ring_lo_features{lo_idx_str}_metadata.json")
                    # elif self.dataset_str == "CSL":
                    #     metadata_filenames.append(f"CSL_{self.r[idx]}-{self.s[idx]}-ring_lo_features{lo_idx_str}_metadata.json")
                    # elif self.dataset_str.endswith("-Prox"):
                    #     metadata_filenames.append(f"{self.h}-Prox_{self.r[idx]}-{self.s[idx]}-ring_lo_features{lo_idx_str}_metadata.json")
                else:
                    vertex_feature_paths.append(osp.join(self.dataset_path, 'results', f'{self.r[idx]}-{self.s[idx]}-ring_SP_features'))
                    if self.dataset_str == "ogbg-molhiv":
                        metadata_filenames.append(f"MOLHIV_{self.r[idx]}-{self.s[idx]}-ring_SP_features_metadata.json")
                    elif self.dataset_str == "ogbg-ppa":
                        metadata_filenames.append(f"PPA_{self.r[idx]}-{self.s[idx]}-ring_SP_features_metadata.json")
                    elif self.dataset_str == "CSL":
                        metadata_filenames.append(f"CSL_{self.r[idx]}-{self.s[idx]}-ring_SP_features_metadata.json")
                    elif self.dataset_str.endswith("-Prox"):
                        metadata_filenames.append(f"{self.h}-Prox_{self.r[idx]}-{self.s[idx]}-ring_SP_features_metadata.json")

        elif self.is_vertex_sp_features:
            num_experiments = len(self.num_clusters) * len(self.pca_dims) * len(self.min_cluster_sizes)
            if self.use_gpnn:
                num_experiments *= len(self.gpnn_channels) * len(self.gpnn_layers)
            vertex_feature_paths.append(osp.join(self.dataset_path, 'results', f'vertex_SP_features'))
            if self.dataset_str == "ogbg-molhiv":
                metadata_filenames.append(f"MOLHIV_vertex_SP_features_metadata.json")
            elif self.dataset_str == "ogbg-ppa":
                metadata_filenames.append(f"PPA_vertex_SP_features_metadata.json")
            elif self.dataset_str == "CSL":
                metadata_filenames.append(f"CSL_vertex_SP_features_metadata.json")
            elif self.dataset_str.endswith("-Prox"):
                metadata_filenames.append(f"{self.h}-Prox_vertex_SP_features_metadata.json")

        else:
            raise ValueError("Invalid vertex feature identifiers")
        
        data["num_experiments"] = num_experiments

        data["model"] = {}

        # The best hyperparameters
        data["res"] = {}
        data["res"]["best_avg_val_perf"] = -1.0
        data["res"]["best_experiment_idx"] = -1
        data["res"]["best_features_path"] = ""
        data["res"]["best_num_clusters"] = -1
        data["res"]["best_pca_dim"] = -1
        data["res"]["best_min_cluster_size"] = -1
        # Set in the parent method
        data["res"]["val_perf"] = {}
        data["res"]["val_perf"]["mean"] = -1.0
        data["res"]["val_perf"]["std"] = -1.0

        data["res"]["test_perf"] = {}
        data["res"]["test_perf"]["mean"] = -1.0
        data["res"]["test_perf"]["std"] = -1.0

        # Stores the data for a single run
        data["experiment_idx"] = {}

        # Stores the average time used for computing a split, a rerun and an epoch (averaged over all total runs (including all folds))
        data["times"] = {}
        data["times"]["experiment_avg"] = -1.0
        data["times"]["cluster_avg"] = -1.0
        data["times"]["enhance_cluster_id_avg"] = -1.0
        data["times"]["rerun_avg"] = -1.0
        data["times"]["epoch_avg"] = -1.0

        # iterate over the hyperparameters for this step
        best_avg_val_perf = 0.0
        best_features_path = ""
        best_features_metadata_filename = ""
        best_clustering_metadata_path = ""
        best_num_clusters = 0
        best_pca_dim = 0
        best_min_cluster_size = 0
        best_val_perf_experiment_idx = 0
        best_max_num_clusters = 0
        best_gpnn_layers = 0
        best_gpnn_channels = 0

        best_val_perfs = []
        best_test_perfs = []

        avg_experiment_time = 0.0
        avg_rerun_time_overall = 0.0
        avg_epoch_time_overall = 0.0
        avg_time_cluster_overall = 0.0
        avg_time_enhance_cluster_id_overall = 0.0

        cur_experiment_idx = 0
        for n_cluster in self.num_clusters:
            for pca_d in self.pca_dims:
                for min_cluster_size in self.min_cluster_sizes:
                    for path_idx, vertex_feature_path in enumerate(vertex_feature_paths):
                        if self.use_gpnn:
                            for gpnn_l in self.gpnn_layers:
                                for gpnn_c in self.gpnn_channels:
                                    experiment_start = time.time()

                                    data["experiment_idx"][cur_experiment_idx] = {}
                                    data["experiment_idx"][cur_experiment_idx]["avg_val_perf"] = -1.0
                                    data["experiment_idx"][cur_experiment_idx]["vertex_feature_path"] = vertex_feature_path

                                    vertex_feature_metadata = util.read_metadata_file(osp.join(self.root_path, vertex_feature_path, metadata_filenames[path_idx]))

                                    # Load dataset and split
                                    dataset = PygGraphPropPredDataset(name = self.dataset_str, root = osp.join(self.root_path, self.dataset_path))

                                    split_idx = dataset.get_idx_split()
                                    train_indices = split_idx["train"]
                                    val_indices = split_idx["valid"]
                                    test_indices = split_idx["test"]

                                    split_prop = { "desc" : f"fixed_{self.dataset_str}", "split_mode" : "fixed", "num_samples" : { "train" : train_indices.size()[0], "val" : val_indices.size()[0], "test" : test_indices.size()[0] }}

                                    evaluator = Evaluator(self.dataset_str)

                                    desc_str = f'{self.dataset_str}_enhanced_{self.model_str}_cluster_exp-{cur_experiment_idx}/{num_experiments}'

                                    # data_folds stores the data for a single fold
                                    data_folds = {}
                                    data_folds["avg_val_perf"] = -1.0
                                    data_folds["split_prop"] = split_prop

                                    # Generate clustering
                                    clusterer.reset_parameters_and_metadata()
                                    feature_vector_database_path = vertex_feature_metadata["result_prop"]["path"]
                                    dataset_desc = vertex_feature_metadata["dataset_prop"]["desc"]
                                    clusterer.load_dataset_from_svmlight(path = feature_vector_database_path, dtype = 'float64', dataset_desc = dataset_desc, split = train_indices, split_prop = split_prop, normalize = self.normalize_features)

                                    # PCA
                                    working_path = osp.join(vertex_feature_path, 'cluster-gnn')

                                    if pca_d > 0 and pca_d < clusterer.dataset.shape[1]:
                                        pca_filename = f'{pca_d}_dim_pca.pkl'
                                        clusterer.generate_pca(target_dimensions = pca_d, write_pca_path = working_path, write_pca_filename = pca_filename)
                                        clusterer.apply_pca_to_dataset()

                                    # k-means
                                    if pca_d > 0 and pca_d < clusterer.dataset.shape[1]:
                                        centroids_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_{pca_d}-pca_centroids.txt'
                                    else:
                                        centroids_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_centroids.txt'

                                    _, centroids, _, clustering_time = clusterer.mini_batch_k_means(n_clusters = n_cluster, min_cluster_size = min_cluster_size, batch_size = constants.mbk_batch_size, n_init = constants.mbk_n_init, max_no_improvement = constants.mbk_max_no_improvement, max_iter = constants.mbk_max_iter)

                                    clusterer.write_centroids_or_medoids(points = centroids, path = working_path, filename = centroids_filename)
                                    avg_time_cluster_overall += clustering_time

                                    num_clusters = centroids.shape[0]

                                    if pca_d > 0 and pca_d < clusterer.dataset.shape[1]:
                                        metadata_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_{pca_d}-pca_cluster_metadata.json'
                                    else:
                                        metadata_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_cluster_metadata.json'

                                    clusterer.write_metadata(path = working_path, filename = metadata_filename)
                                    cluster_metadata = clusterer.get_metadata()
                                    cluster_metadata_path = osp.join(working_path, metadata_filename)

                                    data_folds["cluster_metadata_path"] = cluster_metadata_path

                                    # generate the gnn
                                    self.gnn.set_dataset_parameters(num_classes = dataset.num_tasks, num_features = dataset.num_features, num_clusters = num_clusters)
                                    if self.use_gpnn:
                                        self.load_enhanced_gnn(num_gpnn_layers = gpnn_l, gpnn_channels = gpnn_c, base_gnn_str = self.base_model, num_gnn_layers = n_layers, gnn_hidden_channels = hidden_channels, lr = lr)
                                    else:
                                        self.load_enhanced_gnn(hidden_channels = hidden_channels, num_layers = n_layers, lr = lr)
                                    data["experiment_idx"][cur_experiment_idx]["model"] = self.gnn.get_metadata()

                                    # enhance dataset
                                    dataset, add_cluster_id_time = gnn_utils.include_cluster_id_feature_transform(dataset = dataset, absolute_path_prefix = self.root_path, cluster_metadata = cluster_metadata, vertex_feature_metadata = vertex_feature_metadata)
                                    # dataset, add_cluster_id_time = gnn_utils.include_cluster_id_feature_transform(dataset = dataset, absolute_path_prefix = self.root_path, feature_metadata = vertex_feature_metadata, cluster_metadata = cluster_metadata)

                                    rerun_data = {}

                                    avg_rerun_time = 0.0
                                    avg_rerun_epoch_time = 0.0

                                    test_perfs = []
                                    val_perfs = []

                                    for cur_rerun in range(self.num_reruns):

                                        start_rerun = time.time()

                                        rerun_data[cur_rerun] = {}
                                        rerun_data[cur_rerun]["best_val_perf"] = -1.0
                                        rerun_data[cur_rerun]["best_val_perf_epoch_idx"] = -1
                                        rerun_data[cur_rerun]["epoch"] = {}

                                        self.gnn.model.reset_parameters()

                                        # We shuffle the training data for training
                                        train_loader = DataLoader(dataset = dataset[train_indices], batch_size = s_batch, shuffle = True, num_workers = constants.num_workers)
                                        val_loader = DataLoader(dataset = dataset[val_indices], batch_size = s_batch, shuffle = False, num_workers = constants.num_workers)
                                        test_loader = DataLoader(dataset = dataset[test_indices], batch_size = s_batch, shuffle = False, num_workers = constants.num_workers)

                                        best_rerun_val_perf = -1.0
                                        best_rerun_test_perf = -1.0
                                        best_rerun_val_epoch = 0

                                        avg_epoch_time = 0.0

                                        patience = 0

                                        stop = False

                                        total_num_epoch = 0

                                        for epoch in range(n_epoch):

                                            if stop:
                                                break

                                            t0 = time.time()

                                            # rerun_data[cur_rerun]["epoch"][epoch] = {}

                                            # train the gnn
                                            self.train_ogb(gnn = self.gnn, loader = train_loader, desc = f'{desc_str}_rerun-{cur_rerun}/{self.num_reruns}_epoch-{epoch}/{n_epoch}_train')

                                            # evaluate the results
                                            train_perf = self.eval_ogb(gnn = self.gnn, loader = train_loader, evaluator = evaluator, desc = f'{desc_str}_rerun-{cur_rerun}/{self.num_reruns}_epoch-{epoch}/{n_epoch}_train_eval')[dataset.eval_metric].item()
                                            val_perf = self.eval_ogb(gnn = self.gnn, loader = val_loader, evaluator = evaluator, desc = f'{desc_str}_rerun-{cur_rerun}/{self.num_reruns}_epoch-{epoch}/{n_epoch}_val_eval')[dataset.eval_metric].item()
                                            test_perf = self.eval_ogb(gnn = self.gnn, loader = test_loader, evaluator = evaluator, desc = f'{desc_str}_rerun-{cur_rerun}/{self.num_reruns}_epoch-{epoch}/{n_epoch}_test_eval')[dataset.eval_metric].item()

                                            # rerun_data[cur_rerun]["epoch"][epoch]["perf"] = {'Train': train_perf, 'Validation': val_perf, 'Test': test_perf}

                                            if best_rerun_val_perf < val_perf:
                                                best_rerun_val_perf = val_perf
                                                best_rerun_val_epoch = epoch
                                                best_rerun_test_perf = test_perf
                                                patience = 0
                                            else:
                                                patience += 1
                                                if patience >= self.max_patience:
                                                    stop = True
                                                    # print(f"early stop: {total_num_epoch + 1} epochs")

                                            total_num_epoch += 1
                                            time_epoch = time.time() - t0
                                            # rerun_data[cur_rerun]["epoch"][epoch]["time"] = time_epoch
                                            avg_epoch_time += time_epoch
                                        
                                        avg_epoch_time /= total_num_epoch
                                        avg_rerun_epoch_time += avg_epoch_time

                                        rerun_data[cur_rerun]["best_val_perf"] = best_rerun_val_perf
                                        rerun_data[cur_rerun]["best_val_perf_epoch_idx"] = best_rerun_val_epoch

                                        val_perfs.append(best_rerun_val_perf)
                                        test_perfs.append(best_rerun_test_perf)
                                        avg_rerun_time += time.time() - start_rerun

                                    avg_rerun_time /= self.num_reruns
                                    avg_rerun_epoch_time /= self.num_reruns

                                    mean_val_perf = torch.tensor(val_perfs).mean().item()

                                    # store data
                                    data_folds["rerun"] = rerun_data
                                    
                                    data["experiment_idx"][cur_experiment_idx]["avg_val_perf"] = mean_val_perf
                                    data["experiment_idx"][cur_experiment_idx].update(data_folds)

                                    # These hyperparameters yield the best results so far
                                    if best_avg_val_perf < mean_val_perf:
                                        best_avg_val_perf = mean_val_perf
                                        best_features_path = vertex_feature_path
                                        best_features_metadata_filename = metadata_filenames[path_idx]
                                        best_min_cluster_size = min_cluster_size
                                        best_pca_dim = pca_d
                                        best_num_clusters = n_cluster
                                        best_clustering_metadata_path = cluster_metadata_path
                                        best_val_perf_experiment_idx = cur_experiment_idx
                                        best_val_perfs = val_perfs
                                        best_test_perfs = test_perfs
                                        best_max_num_clusters = num_clusters
                                        best_gpnn_layers = gpnn_l
                                        best_gpnn_channels = gpnn_c
                                    
                                    cur_experiment_idx += 1

                                    avg_rerun_time_overall += avg_rerun_time
                                    avg_epoch_time_overall += avg_rerun_epoch_time
                                    avg_time_cluster_overall += clustering_time
                                    avg_time_enhance_cluster_id_overall += add_cluster_id_time
                                    avg_experiment_time += time.time() - experiment_start
                        else:
                            experiment_start = time.time()

                            data["experiment_idx"][cur_experiment_idx] = {}
                            data["experiment_idx"][cur_experiment_idx]["avg_val_perf"] = -1.0
                            data["experiment_idx"][cur_experiment_idx]["vertex_feature_path"] = vertex_feature_path

                            vertex_feature_metadata = util.read_metadata_file(osp.join(self.root_path, vertex_feature_path, metadata_filenames[path_idx]))

                            # Load dataset and split
                            dataset = PygGraphPropPredDataset(name = self.dataset_str, root = osp.join(self.root_path, self.dataset_path))

                            split_idx = dataset.get_idx_split()
                            train_indices = split_idx["train"]
                            val_indices = split_idx["valid"]
                            test_indices = split_idx["test"]

                            split_prop = { "desc" : f"fixed_{self.dataset_str}", "split_mode" : "fixed", "num_samples" : { "train" : train_indices.size()[0], "val" : val_indices.size()[0], "test" : test_indices.size()[0] }}

                            evaluator = Evaluator(self.dataset_str)

                            desc_str = f'{self.dataset_str}_enhanced_{self.model_str}_cluster_exp-{cur_experiment_idx}/{num_experiments}'

                            # data_folds stores the data for a single fold
                            data_folds = {}
                            data_folds["avg_val_perf"] = -1.0
                            data_folds["split_prop"] = split_prop

                            # Generate clustering
                            clusterer.reset_parameters_and_metadata()
                            feature_vector_database_path = vertex_feature_metadata["result_prop"]["path"]
                            dataset_desc = vertex_feature_metadata["dataset_prop"]["desc"]
                            clusterer.load_dataset_from_svmlight(path = feature_vector_database_path, dtype = 'float64', dataset_desc = dataset_desc, split = train_indices, split_prop = split_prop, normalize = self.normalize_features)

                            # PCA
                            working_path = osp.join(vertex_feature_path, 'cluster-gnn')

                            if pca_d > 0 and pca_d < clusterer.dataset.shape[1]:
                                pca_filename = f'{pca_d}_dim_pca.pkl'
                                clusterer.generate_pca(target_dimensions = pca_d, write_pca_path = working_path, write_pca_filename = pca_filename)
                                clusterer.apply_pca_to_dataset()

                            # k-means
                            if pca_d > 0 and pca_d < clusterer.dataset.shape[1]:
                                centroids_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_{pca_d}-pca_centroids.txt'
                            else:
                                centroids_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_centroids.txt'

                            _, centroids, _, clustering_time = clusterer.mini_batch_k_means(n_clusters = n_cluster, min_cluster_size = min_cluster_size, batch_size = constants.mbk_batch_size, n_init = constants.mbk_n_init, max_no_improvement = constants.mbk_max_no_improvement, max_iter = constants.mbk_max_iter)

                            clusterer.write_centroids_or_medoids(points = centroids, path = working_path, filename = centroids_filename)
                            avg_time_cluster_overall += clustering_time

                            num_clusters = centroids.shape[0]

                            if pca_d > 0 and pca_d < clusterer.dataset.shape[1]:
                                metadata_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_{pca_d}-pca_cluster_metadata.json'
                            else:
                                metadata_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_cluster_metadata.json'

                            clusterer.write_metadata(path = working_path, filename = metadata_filename)
                            cluster_metadata = clusterer.get_metadata()
                            cluster_metadata_path = osp.join(working_path, metadata_filename)

                            data_folds["cluster_metadata_path"] = cluster_metadata_path

                            # generate the gnn
                            self.gnn.set_dataset_parameters(num_classes = dataset.num_tasks, num_features = dataset.num_features, num_clusters = num_clusters)
                            self.load_enhanced_gnn(hidden_channels = hidden_channels, num_layers = n_layers, lr = lr)
                            data["experiment_idx"][cur_experiment_idx]["model"] = self.gnn.get_metadata()

                            # enhance dataset
                            dataset, add_cluster_id_time = gnn_utils.include_cluster_id_feature_transform(dataset = dataset, absolute_path_prefix = self.root_path, cluster_metadata = cluster_metadata, vertex_feature_metadata = vertex_feature_metadata)
                            # dataset, add_cluster_id_time = gnn_utils.include_cluster_id_feature_transform(dataset = dataset, absolute_path_prefix = self.root_path, feature_metadata = vertex_feature_metadata, cluster_metadata = cluster_metadata)

                            rerun_data = {}

                            avg_rerun_time = 0.0
                            avg_rerun_epoch_time = 0.0

                            test_perfs = []
                            val_perfs = []

                            for cur_rerun in range(self.num_reruns):

                                start_rerun = time.time()

                                rerun_data[cur_rerun] = {}
                                rerun_data[cur_rerun]["best_val_perf"] = -1.0
                                rerun_data[cur_rerun]["best_val_perf_epoch_idx"] = -1
                                rerun_data[cur_rerun]["epoch"] = {}

                                self.gnn.model.reset_parameters()

                                # We shuffle the training data for training
                                train_loader = DataLoader(dataset = dataset[train_indices], batch_size = s_batch, shuffle = True, num_workers = constants.num_workers)
                                val_loader = DataLoader(dataset = dataset[val_indices], batch_size = s_batch, shuffle = False, num_workers = constants.num_workers)
                                test_loader = DataLoader(dataset = dataset[test_indices], batch_size = s_batch, shuffle = False, num_workers = constants.num_workers)

                                best_rerun_val_perf = -1.0
                                best_rerun_test_perf = -1.0
                                best_rerun_val_epoch = 0

                                avg_epoch_time = 0.0

                                patience = 0

                                stop = False

                                total_num_epoch = 0

                                for epoch in range(n_epoch):

                                    if stop:
                                        break

                                    t0 = time.time()

                                    # rerun_data[cur_rerun]["epoch"][epoch] = {}

                                    # train the gnn
                                    self.train_ogb(gnn = self.gnn, loader = train_loader, desc = f'{desc_str}_rerun-{cur_rerun}/{self.num_reruns}_epoch-{epoch}/{n_epoch}_train')

                                    # evaluate the results
                                    train_perf = self.eval_ogb(gnn = self.gnn, loader = train_loader, evaluator = evaluator, desc = f'{desc_str}_rerun-{cur_rerun}/{self.num_reruns}_epoch-{epoch}/{n_epoch}_train_eval')[dataset.eval_metric].item()
                                    val_perf = self.eval_ogb(gnn = self.gnn, loader = val_loader, evaluator = evaluator, desc = f'{desc_str}_rerun-{cur_rerun}/{self.num_reruns}_epoch-{epoch}/{n_epoch}_val_eval')[dataset.eval_metric].item()
                                    test_perf = self.eval_ogb(gnn = self.gnn, loader = test_loader, evaluator = evaluator, desc = f'{desc_str}_rerun-{cur_rerun}/{self.num_reruns}_epoch-{epoch}/{n_epoch}_test_eval')[dataset.eval_metric].item()

                                    # rerun_data[cur_rerun]["epoch"][epoch]["perf"] = {'Train': train_perf, 'Validation': val_perf, 'Test': test_perf}

                                    if best_rerun_val_perf < val_perf:
                                        best_rerun_val_perf = val_perf
                                        best_rerun_val_epoch = epoch
                                        best_rerun_test_perf = test_perf
                                        patience = 0
                                    else:
                                        patience += 1
                                        if patience >= self.max_patience:
                                            stop = True
                                            # print(f"early stop: {total_num_epoch + 1} epochs")

                                    total_num_epoch += 1
                                    time_epoch = time.time() - t0
                                    # rerun_data[cur_rerun]["epoch"][epoch]["time"] = time_epoch
                                    avg_epoch_time += time_epoch
                                
                                avg_epoch_time /= total_num_epoch
                                avg_rerun_epoch_time += avg_epoch_time

                                rerun_data[cur_rerun]["best_val_perf"] = best_rerun_val_perf
                                rerun_data[cur_rerun]["best_val_perf_epoch_idx"] = best_rerun_val_epoch

                                val_perfs.append(best_rerun_val_perf)
                                test_perfs.append(best_rerun_test_perf)
                                avg_rerun_time += time.time() - start_rerun

                            avg_rerun_time /= self.num_reruns
                            avg_rerun_epoch_time /= self.num_reruns

                            mean_val_perf = torch.tensor(val_perfs).mean().item()

                            # store data
                            data_folds["rerun"] = rerun_data
                            
                            data["experiment_idx"][cur_experiment_idx]["avg_val_perf"] = mean_val_perf
                            data["experiment_idx"][cur_experiment_idx].update(data_folds)

                            # These hyperparameters yield the best results so far
                            if best_avg_val_perf < mean_val_perf:
                                best_avg_val_perf = mean_val_perf
                                best_features_path = vertex_feature_path
                                best_features_metadata_filename = metadata_filenames[path_idx]
                                best_min_cluster_size = min_cluster_size
                                best_pca_dim = pca_d
                                best_num_clusters = n_cluster
                                best_clustering_metadata_path = cluster_metadata_path
                                best_val_perf_experiment_idx = cur_experiment_idx
                                best_val_perfs = val_perfs
                                best_test_perfs = test_perfs
                                best_max_num_clusters = num_clusters
                            
                            cur_experiment_idx += 1

                            avg_rerun_time_overall += avg_rerun_time
                            avg_epoch_time_overall += avg_rerun_epoch_time
                            avg_time_cluster_overall += clustering_time
                            avg_time_enhance_cluster_id_overall += add_cluster_id_time
                            avg_experiment_time += time.time() - experiment_start

        # Store results
        data["res"]["best_avg_val_perf"] = best_avg_val_perf
        data["res"]["best_experiment_idx"] = best_val_perf_experiment_idx
        data["res"]["best_features_path"] = best_features_path
        data["res"]["best_feature_metadata_filename"] = best_features_metadata_filename
        data["res"]["best_clustering_metadata_path"] = best_clustering_metadata_path
        data["res"]["best_num_clusters"] = best_num_clusters
        data["res"]["best_pca_dim"] = best_pca_dim
        data["res"]["best_min_cluster_size"] = best_min_cluster_size
        data["res"]["max_num_clusters"] = best_max_num_clusters
        if self.use_gpnn:
            data["res"]["best_num_gpnn_layers"] = best_gpnn_layers
            data["res"]["best_gpnn_channels"] = best_gpnn_channels

        avg_experiment_time /= num_experiments
        avg_rerun_time_overall /= num_experiments
        avg_epoch_time_overall /= num_experiments
        avg_time_cluster_overall /= num_experiments
        avg_time_enhance_cluster_id_overall /= num_experiments

        data["times"]["experiment_avg"] = avg_experiment_time
        data["times"]["cluster_avg"] = avg_time_cluster_overall
        data["times"]["enhance_cluster_id_avg"] = avg_time_enhance_cluster_id_overall
        data["times"]["rerun_avg"] = avg_rerun_time_overall
        data["times"]["epoch_avg"] = avg_epoch_time_overall

        return data, best_val_perfs, best_test_perfs, best_features_path, best_features_metadata_filename, best_clustering_metadata_path, best_max_num_clusters, best_num_clusters, best_pca_dim, best_min_cluster_size, best_gpnn_layers, best_gpnn_channels


    # Schedules multiple experiments
    def run_csl_prox_experiments(self):
        
        t0 = time.time()

        print(f'---   Starting experiments on {self.dataset_str}   ---')

        data = {}
        data["dataset"] = self.dataset_str
        data["loss_func"] = "cross_entropy"
        data["metric"] = "acc"
        data["overall_time"] = -1.0
        data["res"] = {}

        loss_func = F.cross_entropy

        if self.dataset_str in ['NCI1', 'ENZYMES', 'PROTEINS', 'DD']:
            # TU dataset
            dataset = TUDataset(name = self.dataset_str, root = osp.join(self.root_path, self.dataset_path), use_node_attr = False)
            splits = util.generate_tu_splits(dataset = dataset, dataset_path = self.dataset_path, root_path = self.root_path)
        elif self.dataset_str == 'CSL':
            dataset = CSL_Dataset(root = osp.join(self.root_path, self.dataset_path))
            splits = dataset.gen_data_splits()
        elif self.dataset_str.endswith('-Prox'):
            dataset = ProximityDataset(root = osp.join(self.root_path, self.dataset_path), h = self.h)
            splits = dataset.gen_data_splits()

        # Optimize GNN hyperparameters without clustering first
        print('---   Optimizing classical GNN hyperparameters   ---')
        hyperparameter_opt_data, best_val_accs, best_test_accs, best_classic_n_layers, best_classic_n_hidden_channels, best_classic_s_batch, best_classic_n_epoch, best_classic_lr = self.run_tu_csl_prox_gnn_hyperparameter_optimization(splits = splits, loss_func = loss_func, classic_gnn = True)

        val_accs = torch.tensor(best_val_accs)
        test_accs = torch.tensor(best_test_accs)
        val_mean = val_accs.mean().item()
        if val_accs.size()[0] == 1:
            val_std = 0.0
        else:
            val_std = val_accs.std().item()
        test_mean = test_accs.mean().item()
        if test_accs.size()[0] == 1:
            test_std = 0.0
        else:
            test_std = test_accs.std().item()

        data["classic_gnn_hyperparameter_opt"] = hyperparameter_opt_data

        data["classic_gnn_hyperparameter_opt"]["res"]["val_acc"]["mean"] = val_mean
        data["classic_gnn_hyperparameter_opt"]["res"]["val_acc"]["std"] = val_std

        data["classic_gnn_hyperparameter_opt"]["res"]["test_acc"]["mean"] = test_mean
        data["classic_gnn_hyperparameter_opt"]["res"]["test_acc"]["std"] = test_std

        print(f'---   Val acc: mean: {val_mean}; std: {val_std}   ---')
        print(f'---   Test acc: mean: {test_mean}; std: {test_std}   ---')

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
        cluster_hyperparameter_opt_data, best_val_accs, best_test_accs, best_features_path, best_feature_metadata_filename, best_clustering_paths, max_num_clusters, best_num_clusters, best_pca_dim, best_min_cluster_size, best_gpnn_layers, best_gpnn_channels = self.run_csl_prox_enhanced_gnn_cluster_hyperparameter_optimization(clusterer = clusterer, n_layers = best_classic_n_layers, 
                                                                                                                                                       hidden_channels = best_classic_n_hidden_channels, s_batch = best_classic_s_batch,
                                                                                                                                                       n_epoch = best_classic_n_epoch, lr = best_classic_lr, splits = splits, loss_func = loss_func, lo_idx_str = self.lo_idx_str)

        val_accs = torch.tensor(best_val_accs)
        test_accs = torch.tensor(best_test_accs)
        val_mean = val_accs.mean().item()
        if val_accs.size()[0] == 1:
            val_std = 0.0
        else:
            val_std = val_accs.std().item()
        test_mean = test_accs.mean().item()
        if test_accs.size()[0] == 1:
            test_std = 0.0
        else:
            test_std = test_accs.std().item()

        data["clustering_hyperparameter_opt"] = cluster_hyperparameter_opt_data

        data["clustering_hyperparameter_opt"]["res"]["val_acc"]["mean"] = val_mean
        data["clustering_hyperparameter_opt"]["res"]["val_acc"]["std"] = val_std

        data["clustering_hyperparameter_opt"]["res"]["test_acc"]["mean"] = test_mean
        data["clustering_hyperparameter_opt"]["res"]["test_acc"]["std"] = test_std

        print(f'---   Val acc: mean: {val_mean}; std: {val_std}   ---')
        print(f'---   Test acc: mean: {test_mean}; std: {test_std}   ---')

        print('---   Optimizing clustering hyperparameters complete   ---')

        # # Testing
        # best_features_path = osp.join(self.dataset_path, 'results', 'vertex_SP_features')
        # best_clustering_paths = {}
        # for idx in range(5):
        #     best_clustering_paths [idx] = osp.join(best_features_path, 'cluster-gnn', f'{idx}-fold', '4-means_min6140-size_cluster_metadata.json')
        # max_num_clusters = 1

        print('---   Running hyperparameter optimization for final enhanced GNN   ---')
        # Re-train enhanced gnn hyperparameter using clustering hyperparameter
        enhanced_hyperparameter_opt_data, best_val_accs, best_test_accs, best_n_layers, best_n_hidden_channels, best_s_batch, best_n_epoch, best_lr = self.run_tu_csl_prox_gnn_hyperparameter_optimization(splits = splits, loss_func = loss_func, classic_gnn = False, best_clustering_paths = best_clustering_paths, best_num_clusters = max_num_clusters, best_vertex_feature_metadata_path = osp.join(best_features_path, best_feature_metadata_filename), best_gpnn_channels = best_gpnn_channels, best_num_gpnn_layers = best_gpnn_layers)

        data["enhanced_gnn_hyperparameter_opt"] = enhanced_hyperparameter_opt_data

        val_accs = torch.tensor(best_val_accs)
        test_accs = torch.tensor(best_test_accs)
        val_mean = val_accs.mean().item()
        if val_accs.size()[0] == 1:
            val_std = 0.0
        else:
            val_std = val_accs.std().item()
        test_mean = test_accs.mean().item()
        if test_accs.size()[0] == 1:
            test_std = 0.0
        else:
            test_std = test_accs.std().item()

        data["enhanced_gnn_hyperparameter_opt"]["res"]["val_acc"]["mean"] = val_mean
        data["enhanced_gnn_hyperparameter_opt"]["res"]["val_acc"]["std"] = val_std

        data["enhanced_gnn_hyperparameter_opt"]["res"]["test_acc"]["mean"] = test_mean
        data["enhanced_gnn_hyperparameter_opt"]["res"]["test_acc"]["std"] = test_std

        data["res"]["test_acc"] = {}
        data["res"]["test_acc"]["mean"] = test_mean
        data["res"]["test_acc"]["std"] = test_std
        data["res"]["val_acc"] = {}
        data["res"]["val_acc"]["mean"] = val_mean
        data["res"]["val_acc"]["std"] = val_std
        
        data["res"]["hyperparameter"] = {}

        data["res"]["hyperparameter"]["best_num_layers"] = best_n_layers
        data["res"]["hyperparameter"]["best_num_hidden_channels"] = best_n_hidden_channels
        data["res"]["hyperparameter"]["best_batch_size"] = best_s_batch
        data["res"]["hyperparameter"]["best_num_epochs"] = best_n_epoch
        data["res"]["hyperparameter"]["best_lr"] = best_lr

        data["res"]["hyperparameter"]["best_num_clusters"] = best_num_clusters
        data["res"]["hyperparameter"]["max_num_clusters"] = max_num_clusters
        data["res"]["hyperparameter"]["best_pca_dim"] = best_pca_dim
        data["res"]["hyperparameter"]["best_min_cluster_size"] = best_min_cluster_size
        data["res"]["hyperparameter"]["best_num_gpnn_layers"] = best_gpnn_layers
        data["res"]["hyperparameter"]["best_gpnn_channels"] = best_gpnn_channels

        print(f'---   Val acc: mean: {val_mean}; std: {val_std}   ---')
        print(f'---   Test acc: mean: {test_mean}; std: {test_std}   ---')

        print('---   Running hyperparameter optimization for final enhanced GNN completed   ---')

        data["overall_time"] = time.time() - t0

        self.data.update(data)

    def run_tu_csl_prox_gnn_hyperparameter_optimization(self, splits: Dict, loss_func, classic_gnn: bool = True, best_clustering_paths: Optional[Dict] = None, best_num_clusters: Optional[int] = None, best_vertex_feature_metadata_path: Optional[str] = None, best_num_gpnn_layers: Optional[int] = None, best_gpnn_channels: Optional[int] = None) -> Tuple[Dict, GNN_Manager, int, int, int, int, float]:
        data = {}

        data["num_experiments"] = (len(self.num_layers) * len(self.hidden_channels) * len(self.batch_sizes) * len(self.num_epochs) * len(self.lrs))

        # Will be overwritten if training enhanced gnn hyperparameters
        if self.dataset_str == 'CSL':
            dataset = CSL_Dataset(root = osp.join(self.root_path, self.dataset_path))
        elif self.dataset_str.endswith('-Prox'):
            dataset = ProximityDataset(root = osp.join(self.root_path, self.dataset_path), h = self.h)
        else:
            dataset = TUDataset(name = self.dataset_str, root = osp.join(self.root_path, self.dataset_path), use_node_attr = False)

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
                                if self.use_gpnn:
                                    self.load_enhanced_gnn(num_gpnn_layers = best_num_gpnn_layers, gpnn_channels = best_gpnn_channels, base_gnn_str = self.base_model, 
                                                           num_gnn_layers = n_layers, gnn_hidden_channels = n_hidden_channels)
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

                                if self.dataset_str.endswith("-Prox"):
                                    split_prop = { "desc" : f"fixed_{self.h}-Prox", "split_mode" : "fixed", "num_samples" : { "train" : len(train_indices), "val" : len(val_indices), "test" : len(test_indices) }, "split_idx" : idx}
                                else:
                                    split_prop = { "desc" : f"{len(splits)}-fold_CV", "split_mode" : "CV", "num_samples" : { "train" : len(train_indices), "val" : len(val_indices), "test" : len(test_indices) }, "split_idx" : idx}

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
                                    else:
                                        dataset = TUDataset(root = osp.join(self.root_path, self.dataset_path), name = self.dataset_str, use_node_attr = False)
                                    feature_metadata_path = best_vertex_feature_metadata_path
                                    dataset, _ = gnn_utils.include_cluster_id_feature_transform(dataset = dataset, absolute_path_prefix = self.root_path, vertex_feature_metadata_path = feature_metadata_path, cluster_metadata_path = best_clustering_paths[idx])
                                    # dataset, _ = gnn_utils.include_cluster_id_feature_transform(dataset = dataset, absolute_path_prefix = self.root_path, feature_metadata_path = feature_metadata_path, cluster_metadata_path = best_clustering_paths[idx])

                                # Evaluate a single experiment given the parameters
                                avg_val_acc, val_accs_fold, rerun_data, avg_test_acc, test_accs_fold, avg_time_rerun_fold, avg_time_epoch_fold = self.run_csl_prox_gnn_hyperparameter_experiment_split(dataset = dataset, s_batch = s_batch, n_epoch = n_epoch, train_indices = train_indices, val_indices = val_indices, test_indices = test_indices, loss_func = loss_func)

                                test_accs.extend(test_accs_fold)
                                val_accs.extend(val_accs_fold)

                                # store data
                                data_folds[idx]["avg_val_acc_split"] = avg_val_acc
                                data_folds[idx]["rerun"] = rerun_data

                                avg_val_acc_overall += avg_val_acc
                                avg_time_rerun += avg_time_rerun_fold
                                avg_time_epoch += avg_time_epoch_fold

                                avg_time_fold += time.time() - fold_start
                            
                            avg_time_fold /= len(splits)
                            avg_val_acc_overall /= len(splits)
                            avg_time_rerun /= len(splits)
                            avg_time_epoch /= len(splits)

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
    def run_csl_prox_gnn_hyperparameter_experiment_split(self, dataset: Dataset, s_batch: int, n_epoch: int, train_indices: Tensor, val_indices: Tensor, test_indices: Tensor, loss_func) -> Tuple[float, Dict, float, float, float]:

        # stored data format:
        # foreach rerun: val acc: best, epoch idx of best; epoch data
        # foreach epoch: val loss, train_loss, time
        
        rerun_data = {}

        avg_val_acc = 0.0
        val_accs = []
        test_accs = []
        avg_time_rerun = 0.0
        avg_time_epoch_overall = 0.0
        avg_test_acc = 0.0
        for cur_rerun in range(self.num_reruns):

            start_rerun = time.time()

            rerun_data[cur_rerun] = {}
            rerun_data[cur_rerun]["best_val_acc"] = -1.0
            rerun_data[cur_rerun]["best_val_acc_epoch_idx"] = -1
            # rerun_data[cur_rerun]["epoch"] = {}

            self.gnn.model.reset_parameters()

            # We shuffle the training data for training
            train_loader = DataLoader(dataset = dataset[train_indices], batch_size = s_batch, shuffle = True, num_workers = constants.num_workers)
            val_loader = DataLoader(dataset = dataset[val_indices], batch_size = s_batch, shuffle = False, num_workers = constants.num_workers)
            test_loader = DataLoader(dataset = dataset[test_indices], batch_size = s_batch, shuffle = False, num_workers = constants.num_workers)

            best_val_loss = float("inf")
            best_val_acc = 0.0
            best_val_epoch = 0
            avg_time_epoch = 0.0
            test_acc = -1.0

            stop = False
            total_num_epoch = 0
            patience = 0

            for epoch in range(n_epoch):
                if stop:
                    break

                t0 = time.time()

                # rerun_data[cur_rerun]["epoch"][epoch] = {}

                self.train(gnn = self.gnn, loader = train_loader, loss_func = loss_func)
                val_loss = self.val(gnn = self.gnn, loader = val_loader, loss_func = loss_func)

                # rerun_data[cur_rerun]["epoch"][epoch]["val_loss"] = val_loss
                # rerun_data[cur_rerun]["epoch"][epoch]["train_loss"] = train_loss

                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    best_val_acc = self.test(gnn = self.gnn, loader = val_loader)
                    best_val_epoch = epoch
                    test_acc = self.test(gnn = self.gnn, loader = test_loader)
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.max_patience:
                        stop = True
                        # print(f"early stop: {total_num_epoch + 1} epochs")
                
                total_num_epoch += 1

                time_epoch = time.time() - t0
                # rerun_data[cur_rerun]["epoch"][epoch]["time"] = time_epoch
                avg_time_epoch += time_epoch
            
            avg_time_epoch /= total_num_epoch
            avg_time_epoch_overall += avg_time_epoch
            avg_test_acc += test_acc
            test_accs.append(test_acc)

            rerun_data[cur_rerun]["best_val_acc"] = best_val_acc
            rerun_data[cur_rerun]["best_val_acc_epoch_idx"] = best_val_epoch

            avg_val_acc += best_val_acc
            val_accs.append(best_val_acc)
            avg_time_rerun += time.time() - start_rerun

        avg_val_acc /= self.num_reruns
        avg_time_rerun /= self.num_reruns
        avg_time_epoch_overall /= self.num_reruns
        avg_test_acc /= self.num_reruns

        return avg_val_acc, val_accs, rerun_data, avg_test_acc, test_accs, avg_time_rerun, avg_time_epoch_overall
    
    # Optimizing the hyperparameters used for clustering: num_clusters, pca_dim, min_cluster_size, k/r&s
    def run_csl_prox_enhanced_gnn_cluster_hyperparameter_optimization(self, clusterer: Vertex_Partition_Clustering, n_layers: int, hidden_channels: int, s_batch: int, n_epoch: int, lr: float, splits: Dict, loss_func, lo_idx_str: str = "") -> Tuple[Dict, str, Dict, int, int, int]:

        data = {}
        num_experiments = -1

        # A list of paths to all the vertex feature directories that should be considered for hyperparameter optimization
        vertex_feature_paths = []
        metadata_filenames = []

        # decide whether k-disks, r-s-rings or vertex_sp_features are used.
        if self.k is not None and len(self.k) > 0:
            # Use k-disks
            num_experiments = len(self.num_clusters) * len(self.pca_dims) * len(self.k) * len(self.min_cluster_sizes)
            if self.use_gpnn:
                num_experiments *= len(self.gpnn_channels) * len(self.gpnn_layers)

            for k in self.k:
                if self.is_lovasz_feature:
                    vertex_feature_paths.append(osp.join(self.dataset_path, 'results', f'{k}-disk_lo_features'))
                    if self.dataset_str.endswith("-Prox"):
                        metadata_filenames.append(f"{self.h}-Prox_{k}-disk_lo_features{lo_idx_str}_metadata.json")
                    else:
                        metadata_filenames.append(f"{self.dataset_str}_{k}-disk_lo_features{lo_idx_str}_metadata.json")

                else:
                    vertex_feature_paths.append(osp.join(self.dataset_path, 'results', f'{k}-disk_SP_features'))
                        
                    if self.dataset_str.endswith("-Prox"):
                        metadata_filenames.append(f"{self.h}-Prox_{k}-disk_SP_features_metadata.json")
                    else:
                        metadata_filenames.append(f"{self.dataset_str}_{k}-disk_SP_features_metadata.json")

        elif self.r is not None and self.s is not None and len(self.r) > 0 and len(self.s) == len(self.r):
            # Use r-s-rings
            num_experiments = len(self.num_clusters) * len(self.pca_dims) * len(self.min_cluster_sizes) * len(self.r)
            if self.use_gpnn:
                num_experiments *= len(self.gpnn_channels) * len(self.gpnn_layers)

            for idx in range(len(self.r)):
                if self.is_lovasz_feature:
                    vertex_feature_paths.append(osp.join(self.dataset_path, 'results', f'{self.r[idx]}-{self.s[idx]}-ring_lo_features'))
                    if self.dataset_str.endswith("-Prox"):
                        metadata_filenames.append(f"{self.h}-Prox_{self.r[idx]}-{self.s[idx]}-ring_lo_features{lo_idx_str}_metadata.json")
                    else:
                        metadata_filenames.append(f"{self.dataset_str}_{self.r[idx]}-{self.s[idx]}-ring_lo_features{lo_idx_str}_metadata.json")
                else:
                    vertex_feature_paths.append(osp.join(self.dataset_path, 'results', f'{self.r[idx]}-{self.s[idx]}-ring_SP_features'))
                    if self.dataset_str.endswith("-Prox"):
                        metadata_filenames.append(f"{self.h}-Prox_{self.r[idx]}-{self.s[idx]}-ring_SP_features_metadata.json")
                    else:
                        metadata_filenames.append(f"{self.dataset_str}_{self.r[idx]}-{self.s[idx]}-ring_SP_features_metadata.json")


        elif self.is_vertex_sp_features:
            num_experiments = len(self.num_clusters) * len(self.pca_dims) * len(self.min_cluster_sizes)
            if self.use_gpnn:
                num_experiments *= len(self.gpnn_channels) * len(self.gpnn_layers)
            
            vertex_feature_paths.append(osp.join(self.dataset_path, 'results', f'vertex_SP_features'))
            if self.dataset_str.endswith("-Prox"):
                metadata_filenames.append(f"{self.h}-Prox_vertex_SP_features_metadata.json")
            else:
                metadata_filenames.append(f"{self.dataset_str}_vertex_SP_features_metadata.json")

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
        data["res"]["best_pca_dim"] = -1
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
        best_features_metadata_filename = ""
        best_clustering_metadata_paths = {} # For each experiment we store the cluster results for each fold
        best_num_clusters = 0
        best_pca_dim = 0
        best_min_cluster_size = 0
        best_val_acc_experiment_idx = 0
        best_max_num_clusters = 0
        best_gpnn_layers = 0
        best_gpnn_channels = 0

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
            for pca_d in self.pca_dims:
                for min_cluster_size in self.min_cluster_sizes:
                    for path_idx, vertex_feature_path in enumerate(vertex_feature_paths):
                        if self.use_gpnn:
                            for gpnn_l in self.gpnn_layers:
                                for gpnn_c in self.gpnn_channels:

                                    # Run the optimization, the different feature datasets are defined by the vertex_feature_path
                                    experiment_start = time.time()

                                    if self.dataset_str == 'CSL':
                                        dataset = CSL_Dataset(root = osp.join(self.root_path, self.dataset_path))
                                    elif self.dataset_str.endswith('-Prox'):
                                        dataset = ProximityDataset(root = osp.join(self.root_path, self.dataset_path), h = self.h)
                                    else:
                                        dataset = TUDataset(root = osp.join(self.root_path, self.dataset_path), name = self.dataset_str, use_node_attr = False)

                                    vertex_feature_metadata = util.read_metadata_file(osp.join(self.root_path, vertex_feature_path, metadata_filenames[path_idx]))
                                    # set up clusterer
                                    feature_vector_database_path = vertex_feature_metadata["result_prop"]["path"]
                                    dataset_desc = vertex_feature_metadata["dataset_prop"]["desc"]

                                    clusterer.load_dataset_from_svmlight(path = feature_vector_database_path, dtype = 'float64', dataset_desc = dataset_desc, normalize = self.normalize_features)

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
                                    self.load_enhanced_gnn(gpnn_channels = gpnn_c, num_gpnn_layers = gpnn_l, gnn_hidden_channels= hidden_channels, num_gnn_layers = n_layers, lr = lr)
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
                                            split_prop = { "desc" : f"{len(splits)}-fold_CV", "split_mode" : "CV", "num_samples" : { "train" : len(train_indices), "val" : len(val_indices), "test" : len(test_indices) }, "split_idx" : idx}
                                        elif self.dataset_str.endswith('-Prox'):
                                            split_prop = { "desc" : f"fixed_{self.h}-Prox", "split_mode" : "fixed", "num_samples" : { "train" : len(train_indices), "val" : len(val_indices), "test" : len(test_indices) }, "split_idx" : idx}

                                        data_folds[idx]["split_prop"] = split_prop

                                        train_indices = torch.tensor(train_indices, dtype = torch.long)
                                        val_indices = torch.tensor(val_indices, dtype = torch.long)

                                        # Load the current dataset into the clusterer
                                        clusterer.reset_parameters_and_metadata()
                                        clusterer.set_split(split = train_indices, split_prop = split_prop)
                                        # clusterer.load_dataset_from_svmlight(path = feature_vector_database_path, dtype = 'float64', dataset_desc = dataset_desc, split = train_indices, split_prop = split_prop, normalize = self.normalize_features)

                                        # PCA
                                        working_path = osp.join(vertex_feature_path, 'cluster-gnn', f'{idx}-split')

                                        if pca_d > 0 and pca_d < clusterer.dataset.shape[1]:
                                            pca_filename = f'{pca_d}_dim_pca.pkl'
                                            clusterer.generate_pca(target_dimensions = pca_d, write_pca_path = working_path, write_pca_filename = pca_filename)
                                            clusterer.apply_pca_to_dataset()

                                        # k-means
                                        if pca_d > 0 and pca_d < clusterer.dataset.shape[1]:
                                            centroids_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_{pca_d}-pca_centroids.txt'
                                        else:
                                            centroids_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_centroids.txt'

                                        _, centroids, _, clustering_time = clusterer.mini_batch_k_means(n_clusters = n_cluster, min_cluster_size = min_cluster_size, batch_size = constants.mbk_batch_size, n_init = constants.mbk_n_init, max_no_improvement = constants.mbk_max_no_improvement, max_iter = constants.mbk_max_iter)

                                        clusterer.write_centroids_or_medoids(points = centroids, path = working_path, filename = centroids_filename)
                                        avg_time_cluster += clustering_time

                                        num_clusters = centroids.shape[0]
                                        if num_clusters > max_num_clusters:
                                            max_num_clusters = num_clusters

                                        if pca_d > 0 and pca_d < clusterer.dataset.shape[1]:
                                            metadata_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_{pca_d}-pca_cluster_metadata.json'
                                        else:
                                            metadata_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_cluster_metadata.json'

                                        clusterer.write_metadata(path = working_path, filename = metadata_filename)
                                        cluster_metadata = clusterer.get_metadata()
                                        cluster_metadata_path = osp.join(working_path, metadata_filename)

                                        fold_clustering_metadata_paths[idx] = cluster_metadata_path

                                        data_folds[idx]["cluster_metadata_path"] = cluster_metadata_path

                                        # Create a new dataset enhanced with cluster_ids for GNN training
                                        if self.dataset_str == 'CSL':
                                            dataset = CSL_Dataset(root = osp.join(self.root_path, self.dataset_path))
                                        elif self.dataset_str.endswith('-Prox'):
                                            dataset = ProximityDataset(root = osp.join(self.root_path, self.dataset_path), h = self.h)
                                        else:
                                            dataset = TUDataset(root = osp.join(self.root_path, self.dataset_path), name = self.dataset_str, use_node_attr = False)
                                        dataset, add_cluster_id_time = gnn_utils.include_cluster_id_feature_transform(absolute_path_prefix = self.root_path, dataset = dataset, feature_dataset = clusterer.original_dataset, cluster_metadata = cluster_metadata)
                                        # dataset, add_cluster_id_time = gnn_utils.include_cluster_id_feature_transform(absolute_path_prefix = self.root_path, dataset = dataset, feature_metadata = vertex_feature_metadata, cluster_metadata = cluster_metadata)
                                        avg_time_enhance_cluster_id += add_cluster_id_time

                                        # Evaluate a single experiment given the parameters
                                        avg_val_acc_fold, val_accs_fold, rerun_data, avg_test_acc_fold, test_accs_fold, avg_time_rerun_fold, avg_time_epoch_fold = self.run_csl_prox_gnn_hyperparameter_experiment_split(dataset = dataset, s_batch = s_batch, n_epoch = n_epoch, train_indices = train_indices, val_indices = val_indices, test_indices = test_indices, loss_func = loss_func)

                                        # store data
                                        data_folds[idx]["avg_val_acc_split"] = avg_val_acc_fold
                                        data_folds[idx]["rerun"] = rerun_data

                                        val_accs.extend(val_accs_fold)
                                        test_accs.extend(test_accs_fold)

                                        avg_val_acc += avg_val_acc_fold
                                        avg_time_rerun += avg_time_rerun_fold
                                        avg_time_epoch += avg_time_epoch_fold

                                        avg_time_fold += time.time() - fold_start
                                        
                                    avg_time_fold /= len(splits)
                                    avg_val_acc /= len(splits)
                                    avg_time_rerun /= len(splits)
                                    avg_time_epoch /= len(splits)
                                    avg_time_cluster /= len(splits)
                                    avg_time_enhance_cluster_id /= len(splits)

                                    data["experiment_idx"][cur_experiment_idx]["avg_val_acc"] = avg_val_acc

                                    data["experiment_idx"][cur_experiment_idx]["splits"] = data_folds

                                    # These hyperparameters yield the best results so far
                                    if avg_val_acc > best_avg_val_acc:
                                        best_avg_val_acc = avg_val_acc
                                        best_min_cluster_size = min_cluster_size
                                        best_pca_dim = pca_d
                                        best_num_clusters = n_cluster
                                        best_features_path = vertex_feature_path
                                        best_features_metadata_filename = metadata_filenames[path_idx]
                                        best_val_acc_experiment_idx = cur_experiment_idx
                                        best_clustering_metadata_paths = fold_clustering_metadata_paths
                                        best_max_num_clusters = max_num_clusters
                                        best_val_accs = val_accs
                                        best_test_accs = test_accs
                                        best_gpnn_channels = gpnn_c
                                        best_gpnn_layers = gpnn_l
                                        
                                    cur_experiment_idx += 1

                                    avg_fold_time_overall += avg_time_fold
                                    avg_rerun_time_overall += avg_time_rerun
                                    avg_epoch_time_overall += avg_time_epoch
                                    avg_time_cluster_overall += avg_time_cluster
                                    avg_time_enhance_cluster_id_overall += avg_time_enhance_cluster_id

                                    avg_experiment_time += time.time() - experiment_start
                        else:
                            # Run the optimization, the different feature datasets are defined by the vertex_feature_path
                            experiment_start = time.time()

                            if self.dataset_str == 'CSL':
                                dataset = CSL_Dataset(root = osp.join(self.root_path, self.dataset_path))
                            elif self.dataset_str.endswith('-Prox'):
                                dataset = ProximityDataset(root = osp.join(self.root_path, self.dataset_path), h = self.h)
                            else:
                                dataset = TUDataset(root = osp.join(self.root_path, self.dataset_path), name = self.dataset_str, use_node_attr = False)

                            vertex_feature_metadata = util.read_metadata_file(osp.join(self.root_path, vertex_feature_path, metadata_filenames[path_idx]))
                            # set up clusterer
                            feature_vector_database_path = vertex_feature_metadata["result_prop"]["path"]
                            dataset_desc = vertex_feature_metadata["dataset_prop"]["desc"]

                            clusterer.load_dataset_from_svmlight(path = feature_vector_database_path, dtype = 'float64', dataset_desc = dataset_desc, normalize = self.normalize_features)

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
                                    split_prop = { "desc" : f"{len(splits)}-fold_CV", "split_mode" : "CV", "num_samples" : { "train" : len(train_indices), "val" : len(val_indices), "test" : len(test_indices) }, "split_idx" : idx}
                                elif self.dataset_str.endswith('-Prox'):
                                    split_prop = { "desc" : f"fixed_{self.h}-Prox", "split_mode" : "fixed", "num_samples" : { "train" : len(train_indices), "val" : len(val_indices), "test" : len(test_indices) }, "split_idx" : idx}

                                data_folds[idx]["split_prop"] = split_prop

                                train_indices = torch.tensor(train_indices, dtype = torch.long)
                                val_indices = torch.tensor(val_indices, dtype = torch.long)

                                # Load the current dataset into the clusterer
                                clusterer.reset_parameters_and_metadata()
                                clusterer.set_split(split = train_indices, split_prop = split_prop)
                                # clusterer.load_dataset_from_svmlight(path = feature_vector_database_path, dtype = 'float64', dataset_desc = dataset_desc, split = train_indices, split_prop = split_prop, normalize = self.normalize_features)

                                # PCA
                                working_path = osp.join(vertex_feature_path, 'cluster-gnn', f'{idx}-split')

                                if pca_d > 0 and pca_d < clusterer.dataset.shape[1]:
                                    pca_filename = f'{pca_d}_dim_pca.pkl'
                                    clusterer.generate_pca(target_dimensions = pca_d, write_pca_path = working_path, write_pca_filename = pca_filename)
                                    clusterer.apply_pca_to_dataset()

                                # k-means
                                if pca_d > 0 and pca_d < clusterer.dataset.shape[1]:
                                    centroids_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_{pca_d}-pca_centroids.txt'
                                else:
                                    centroids_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_centroids.txt'

                                _, centroids, _, clustering_time = clusterer.mini_batch_k_means(n_clusters = n_cluster, min_cluster_size = min_cluster_size, batch_size = constants.mbk_batch_size, n_init = constants.mbk_n_init, max_no_improvement = constants.mbk_max_no_improvement, max_iter = constants.mbk_max_iter)

                                clusterer.write_centroids_or_medoids(points = centroids, path = working_path, filename = centroids_filename)
                                avg_time_cluster += clustering_time

                                num_clusters = centroids.shape[0]
                                if num_clusters > max_num_clusters:
                                    max_num_clusters = num_clusters

                                if pca_d > 0 and pca_d < clusterer.dataset.shape[1]:
                                    metadata_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_{pca_d}-pca_cluster_metadata.json'
                                else:
                                    metadata_filename = f'{n_cluster}-means_min-{min_cluster_size}-size_cluster_metadata.json'

                                clusterer.write_metadata(path = working_path, filename = metadata_filename)
                                cluster_metadata = clusterer.get_metadata()
                                cluster_metadata_path = osp.join(working_path, metadata_filename)

                                fold_clustering_metadata_paths[idx] = cluster_metadata_path

                                data_folds[idx]["cluster_metadata_path"] = cluster_metadata_path

                                # Create a new dataset enhanced with cluster_ids for GNN training
                                if self.dataset_str == 'CSL':
                                    dataset = CSL_Dataset(root = osp.join(self.root_path, self.dataset_path))
                                elif self.dataset_str.endswith('-Prox'):
                                    dataset = ProximityDataset(root = osp.join(self.root_path, self.dataset_path), h = self.h)
                                else:
                                    dataset = TUDataset(root = osp.join(self.root_path, self.dataset_path), name = self.dataset_str, use_node_attr = False)
                                dataset, add_cluster_id_time = gnn_utils.include_cluster_id_feature_transform(absolute_path_prefix = self.root_path, dataset = dataset, feature_dataset = clusterer.original_dataset, cluster_metadata = cluster_metadata)
                                # dataset, add_cluster_id_time = gnn_utils.include_cluster_id_feature_transform(absolute_path_prefix = self.root_path, dataset = dataset, feature_metadata = vertex_feature_metadata, cluster_metadata = cluster_metadata)
                                avg_time_enhance_cluster_id += add_cluster_id_time

                                # Evaluate a single experiment given the parameters
                                avg_val_acc_fold, val_accs_fold, rerun_data, avg_test_acc_fold, test_accs_fold, avg_time_rerun_fold, avg_time_epoch_fold = self.run_csl_prox_gnn_hyperparameter_experiment_split(dataset = dataset, s_batch = s_batch, n_epoch = n_epoch, train_indices = train_indices, val_indices = val_indices, test_indices = test_indices, loss_func = loss_func)

                                # store data
                                data_folds[idx]["avg_val_acc_split"] = avg_val_acc_fold
                                data_folds[idx]["rerun"] = rerun_data

                                val_accs.extend(val_accs_fold)
                                test_accs.extend(test_accs_fold)

                                avg_val_acc += avg_val_acc_fold
                                avg_time_rerun += avg_time_rerun_fold
                                avg_time_epoch += avg_time_epoch_fold

                                avg_time_fold += time.time() - fold_start
                                
                            avg_time_fold /= len(splits)
                            avg_val_acc /= len(splits)
                            avg_time_rerun /= len(splits)
                            avg_time_epoch /= len(splits)
                            avg_time_cluster /= len(splits)
                            avg_time_enhance_cluster_id /= len(splits)

                            data["experiment_idx"][cur_experiment_idx]["avg_val_acc"] = avg_val_acc

                            data["experiment_idx"][cur_experiment_idx]["splits"] = data_folds

                            # These hyperparameters yield the best results so far
                            if avg_val_acc > best_avg_val_acc:
                                best_avg_val_acc = avg_val_acc
                                best_min_cluster_size = min_cluster_size
                                best_pca_dim = pca_d
                                best_num_clusters = n_cluster
                                best_features_path = vertex_feature_path
                                best_features_metadata_filename = metadata_filenames[path_idx]
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
        data["res"]["best_feature_metadata_filename"] = best_features_metadata_filename
        data["res"]["best_clustering_metadata_paths"] = best_clustering_metadata_paths
        data["res"]["best_num_clusters"] = best_num_clusters
        data["res"]["best_pca_dim"] = best_pca_dim
        data["res"]["best_min_cluster_size"] = best_min_cluster_size
        data["res"]["max_num_clusters"] = best_max_num_clusters
        if self.use_gpnn:
            data["res"]["best_num_gpnn_layers"] = best_gpnn_layers
            data["res"]["best_gpnn_channels"] = best_gpnn_channels

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

        return data, best_val_accs, best_test_accs, best_features_path, best_features_metadata_filename, best_clustering_metadata_paths, best_max_num_clusters, best_num_clusters, best_pca_dim, best_min_cluster_size, best_gpnn_layers, best_gpnn_channels

    def get_data(self) -> Dict:
        return deepcopy(self.data)

    def train_ogb(self, gnn: GNN_Manager, loader: DataLoader, desc: str) -> float:
        total_loss = 0.0
        gnn.model.train()
        
        if self.dataset_str == 'ogbg-ppa':
            for step, batch in enumerate(loader):
            # for step, batch in enumerate(tqdm(loader, desc = desc)):
                batch = batch.to(gnn.device)

                if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                    pass
                else:
                    pred = gnn.model(batch)
                    gnn.optimizer.zero_grad()

                    loss = self.criterion(pred.to(torch.float32), batch.y.view(-1,))

                    total_loss += loss.item() * batch.num_graphs

                    loss.backward()
                    gnn.optimizer.step()
        elif self.dataset_str == 'ogbg-molhiv':
            for step, batch in enumerate(loader):
            # for step, batch in enumerate(tqdm(loader, desc = desc)):
                batch = batch.to(gnn.device)

                if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                    pass
                else:
                    pred = gnn.model(batch)
                    gnn.optimizer.zero_grad()

                    is_labeled = batch.y == batch.y # ignore NaN targets
                    
                    loss = self.criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])

                    total_loss += loss.item() * batch.num_graphs

                    loss.backward()
                    gnn.optimizer.step()

        return total_loss / len(loader)

    @torch.no_grad()
    def eval_ogb(self, gnn: GNN_Manager, loader: DataLoader, evaluator: Evaluator, desc: str) -> float:
        gnn.model.eval()

        y_true = []
        y_pred = []

        if self.dataset_str == 'ogbg-ppa':
            for step, batch in enumerate(loader):
            # for step, batch in enumerate(tqdm(loader, desc = desc)):
                batch = batch.to(gnn.device)

                if batch.x.shape[0] == 1:
                    pass
                else:
                    with torch.no_grad():
                        pred = gnn.model(batch)
                    
                    y_true.append(batch.y.view(pred.shape).detach().cpu())
                    y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())

            y_true = torch.cat(y_true, dim = 0).numpy()
            y_pred = torch.cat(y_pred, dim = 0).numpy()

            input_dict = {"y_true" : y_true, "y_pred" : y_pred}

            return evaluator.eval(input_dict)

        elif self.dataset_str == 'ogbg-molhiv':
            for step, batch in enumerate(loader):
            # for step, batch in enumerate(tqdm(loader, desc = desc)):
                batch = batch.to(gnn.device)

                if batch.x.shape[0] == 1:
                    pass
                else:
                    with torch.no_grad():
                        pred = gnn.model(batch)

                    y_true.append(batch.y.view(pred.shape).detach().cpu())
                    y_pred.append(pred.detach().cpu())
            
            y_true = torch.cat(y_true, dim = 0).numpy()
            y_pred = torch.cat(y_pred, dim = 0).numpy()

            input_dict = {"y_true" : y_true, "y_pred" : y_pred}

            return evaluator.eval(input_dict)


    # Trains the gnn based on loss_func, returns the average loss per graph
    def train(self, gnn: GNN_Manager, loader: DataLoader, loss_func, desc: Optional[str] = "train") -> float:
        gnn.model.train()

        total_loss = 0.0
        for step, data in enumerate(loader):
        # for step, data in enumerate(tqdm(loader, desc = desc)):
            data = data.to(gnn.device)
            gnn.optimizer.zero_grad()
            loss = loss_func(gnn.model(data), data.y)
            loss.backward()
            total_loss += data.num_graphs * loss.item()
            gnn.optimizer.step()

        return total_loss/len(loader.dataset)

    @torch.no_grad()
    def val(self, gnn: GNN_Manager, loader: DataLoader, loss_func, desc: Optional[str] = "eval") -> float:
        gnn.model.eval()

        total_loss = 0.0
        for step, data in enumerate(loader):
        # for step, data in enumerate(tqdm(loader, desc = desc)):
            data = data.to(gnn.device)
            total_loss += data.num_graphs * loss_func(gnn.model(data), data.y).item()

        return total_loss / len(loader.dataset)
    
    @torch.no_grad()
    def test(self, gnn: GNN_Manager, loader: DataLoader, desc: Optional[str] = "test") -> float:
        gnn.model.eval()

        correct = 0
        for step, data in enumerate(loader):
        # for step, data in enumerate(tqdm(loader, desc = desc)):
            data = data.to(gnn.device)
            pred = gnn.model(data).max(dim = 1)[1]  # gnn.model(data).max(dim = 1) returns a tuple max_vals, max_indices
            correct += pred.eq(data.y).sum().item()

        return correct / len(loader.dataset)
