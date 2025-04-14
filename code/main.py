import os.path as osp
import numpy as np
import os
from typing import Dict, List

import util
import constants

import torch

from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

from experiments import Experiment_Manager
from vertex_partition_feature_generation_main import run_csl, run_proximity, run_molhiv, run_ppa

# Defines an example config file for a run and creates it
def gen_experiment_config_file(root_path: str) -> None:
    path = osp.join(root_path, 'experiments', 'configs')
    filename = 'example_experiment.json'

    config = {}
    config["type"] = "experiment"
    config["title"] = "---   filename of the result (without extension)   ---"
    config["general"] = {}
    config["general"]["seed"] = constants.SEED
    config["general"]["num_reruns"] = constants.num_reruns
    config["general"]["num_k_fold"] = constants.num_k_fold
    config["general"]["k_fold_test_ratio"] = constants.k_fold_test_ratio
    config["general"]["mbk_batch_size"] = constants.mbk_batch_size
    config["general"]["mbk_num_init"] = constants.mbk_n_init
    config["general"]["mbk_max_no_improvement"] = constants.mbk_max_no_improvement
    config["general"]["mbk_max_iter"] = constants.mbk_max_iter
    config["dataset"] = {}
    config["dataset"]["dataset_str"] = "---   'ogbg-molhiv', 'ogbg-ppa', 'CSL' or 'h-Prox' with h = 1,3,5,8,10   ---"
    config["dataset"]["base_model"] = "---   'gin' or 'gcn'   ---"
    config["dataset"]["k"] = ["---   List of k values for k-disks that should be evaluated   ---"]
    config["dataset"]["r"] = ["---   List of r values for r-s-rings that should be evaluated. NOTE: r[idx]-s[idx]-rings will be evaluated   ---"]
    config["dataset"]["s"] = ["---   List of s values for r-s-rings that should be evaluated. NOTE: r[idx]-s[idx]-rings will be evaluated   ---"]
    config["dataset"]["is_vertex_sp_feature"] = False
    config["dataset"]["normalize_vertex_features"] = False
    config["hyperparameters"] = {}
    config["hyperparameters"]["num_clusters"] = ["---   List of the numbers of clusters that will be evaluated   ---"]
    config["hyperparameters"]["lsa_dims"] = ["---   List of the numbers of lsa dimensions that will be evaluated. NOTE: values smaller than 1 mean no dimensionality reduction is performed   ---"]
    config["hyperparameters"]["min_cluster_size"] = ["---   List of the minimum sizes of clusters that will be evaluated   ---"]
    config["hyperparameters"]["num_layers"] = ["---   List of the number of layers of gnns that will be evaluated   ---"]
    config["hyperparameters"]["num_hidden_channels"] = ["---   List of the number hidden dimensions of gnns that will be evaluated   ---"]
    config["hyperparameters"]["num_batch_sizes"] = ["---   Defines the batch sizes of gnns while training   ---"]
    config["hyperparameters"]["num_epochs"] = ["---   List of the number of epochs while training gnns that will be evaluated   ---"]
    config["hyperparameters"]["lrs"] = ["---   List of the learning rates that will be evaluated   ---"]

    util.write_metadata_file(path = path, filename = filename, data = config)

def run_experiment(config: Dict, root_path: str, experiment_idx: int) -> None:

    path = osp.join(root_path, 'experiments', 'results')

    # Parse the config and sanitize the input
    assert config["type"] == "experiment"
    assert "general" in config
    assert "dataset" in config
    assert "hyperparameters" in config

    k = None
    r = None
    s = None
    is_vertex_sp_features = False
    num_clusters = None
    lsa_dims = None
    min_cluster_sizes = None
    num_layers = None
    hidden_channels = None
    batch_sizes = None
    num_epochs = None
    lrs = None
    normalize = False
    dataset_str = ''
    base_model = ''

    config["general"]["mbk_batch_size"] = constants.mbk_batch_size
    config["general"]["mbk_num_init"] = constants.mbk_n_init
    config["general"]["mbk_max_no_improvement"] = constants.mbk_max_no_improvement
    config["general"]["mbk_max_iter"] = constants.mbk_max_iter

    # Parse config
    for key, value in config["general"].items():
        if key == "seed":
            assert isinstance(value, int)
            assert value >= 0
            constants.SEED = value
        elif key == "num_reruns":
            assert isinstance(value, int)
            assert value > 0
            constants.num_reruns = value
        elif key == "num_k_fold":
            assert isinstance(value, int)
            assert value > 0
            constants.num_k_fold = value
        elif key == "k_fold_test_ratio":
            assert isinstance(value, float)
            assert value > 0 and value < 1
            constants.k_fold_test_ratio = value
        elif key == "mbk_batch_size":
            assert isinstance(value, int)
            assert value > 0
            constants.mbk_batch_size = value
        elif key == "mbk_num_init":
            assert isinstance(value, int)
            assert value > 0
            constants.mbk_batch_size = value
        elif key == "mbk_max_no_improvement":
            assert isinstance(value, int)
            assert value > 0
            constants.mbk_max_no_improvement = value
        elif key == "mbk_max_iter":
            assert isinstance(value, int)
            assert value > 0
            constants.mbk_max_iter = value
        else:
            raise ValueError(f'Invalid key {key} in config["general"]')

    for key, value in config["dataset"].items():
        if key == "dataset_str":
            assert isinstance(value, str)
            if value == 'ogbg-molhiv':
                dataset_str = value
            elif value == 'ogbg-ppa':
                dataset_str = value
            elif value == 'CSL':
                dataset_str = value
            elif value.endswith('-Prox'):
                h = int(value[0])
                assert isinstance(h, int) and h in [1,3,5,8,10]
                assert value == f'{h}-Prox'
                dataset_str = value
            else:
                raise ValueError(f'Invalid dataset_str: {value}')
        elif key == "k":
            if len(value) > 0:
                assert [x > 0 for x in value]
                k = value
        elif key == "r":
            if len(value) > 0:
                assert [x > 0 for x in value]
                if s is not None:
                    assert len(s) == len(value)
                    for idx in range(len(s)):
                        assert s[idx] >= value[idx]
                r = value
        elif key == "s":
            if len(value) > 0:
                if r is not None:
                    assert len(r) == len(value)
                    for idx in range(len(r)):
                        assert r[idx] <= value[idx]
                s = value
        elif key == "is_vertex_sp_feature":
            assert isinstance(value, bool)
            is_vertex_sp_features = value
        elif key == "normalize_vertex_features":
            assert isinstance(value, bool)
            normalize = value
        elif key == "base_model":
            assert isinstance(value, str)
            assert value == 'gin' or value == 'gcn'
            base_model = value
        else:
            raise ValueError(f'Invalid key {key} in config["dataset"]')
        
    assert dataset_str is not None and base_model is not None
    assert k is not None or (r is not None and s is not None) or is_vertex_sp_features
    if r is not None:
        assert s is not None
    if s is not None:
        assert r is not None

    for key, value in config["hyperparameters"].items():
        if key == "num_clusters":
            assert len(value) > 0
            assert [x > 0 for x in value]
            num_clusters = value
        elif key == "lsa_dims":
            assert len(value) > 0
            assert [x > 0 for x in value]
            lsa_dims = value
        elif key == "min_cluster_size":
            assert len(value) > 0
            assert [x > 0 for x in value]
            min_cluster_sizes = value
        elif key == "num_layers":
            assert len(value) > 0
            assert [x > 0 for x in value]
            num_layers = value
        elif key == "num_hidden_channels":
            assert len(value) > 0
            assert [x > 0 for x in value]
            hidden_channels = value
        elif key == "num_batch_sizes":
            assert len(value) > 0
            assert [x > 0 for x in value]
            batch_sizes = value
        elif key == "num_epochs":
            assert len(value) > 0
            assert [x > 0 for x in value]
            num_epochs = value
        elif key == "lrs":
            assert len(value) > 0
            assert [x > 0 for x in value]
            lrs = value
        else:
            raise ValueError(f'Invalid key {key} in config["hyperparameters"]')
        

    util.initialize_random_seeds(constants.SEED)

    manager = Experiment_Manager(root_path = root_path)

    try:
        manager.setup_experiments(dataset_str = dataset_str, base_model = base_model, k = k, r = r, s = s, is_vertex_sp_features = is_vertex_sp_features, num_clusters = num_clusters,
                                    lsa_dims = lsa_dims, min_cluster_sizes = min_cluster_sizes, num_layers = num_layers, hidden_channels = hidden_channels, batch_sizes = batch_sizes,
                                    num_epochs = num_epochs, lrs = lrs, normalize_features = normalize)
        
        manager.run_experiments()

        if "title" in config and isinstance(config["title"], str) and len(config["title"]) > 0:
            filename = f'{config["title"]}.json'
        else:
            filename = f'experiment_{experiment_idx}_result.json'

        util.write_metadata_file(data = manager.data, path = path, filename = filename)

    except Exception as e:
        if "title" in config and isinstance(config["title"], str) and len(config["title"]) > 0:
            filename = f'{config["title"]}.json'
        else:
            filename = f'experiment_{experiment_idx}_result.json'
        print(repr(e))
        util.write_metadata_file(data = repr(e), path = path, filename = filename)

# Defines an example config file for a feature generation and creates it
def gen_feature_gen_config_file(root_path: str) -> None:
    path = osp.join(root_path, 'experiments', 'configs')
    filename = 'example_feature_gen.json'

    config = {}
    config["type"] = "feature_gen"
    config["general"] = {}
    config["general"]["seed"] = constants.SEED
    config["general"]["num_processes"] = constants.num_processes
    config["general"]["graph_chunksize"] = constants.graph_chunksize
    config["general"]["vertex_chunksize"] = constants.vertex_chunksize
    config["general"]["vector_buffer_size"] = constants.vector_buffer_size
    config["dataset"] = {}
    config["dataset"]["dataset_str"] = "---   'ogbg-molhiv', 'ogbg-ppa', 'CSL' or 'h-Prox' with h = 1,3,5,8,10   ---"
    config["dataset"]["k"] = ["---   List of k values for k-disks that should be evaluated   ---"]
    config["dataset"]["r"] = ["---   List of r values for r-s-rings that should be evaluated. NOTE: r[idx]-s[idx]-rings will be evaluated   ---"]
    config["dataset"]["s"] = ["---   List of s values for r-s-rings that should be evaluated. NOTE: r[idx]-s[idx]-rings will be evaluated   ---"]
    config["dataset"]["gen_vertex_sp_features"] = False
    config["dataset"]["re_gen_properties"] = True

    util.write_metadata_file(path = path, filename = filename, data = config)

def run_feature_gen(config: Dict, root_path: str) -> None:


    # Parse the config and sanitize the input
    assert config["type"] == "feature_gen"
    assert "general" in config
    assert "dataset" in config

    h = None
    k = None
    r = None
    s = None
    gen_vertex_sp_features = False
    re_gen_properties = True
    dataset_str = ''

    # Parse config
    for key, value in config["general"].items():
        if key == "seed":
            assert isinstance(value, int)
            assert value >= 0
            constants.SEED = value
        elif key == "num_processes":
            assert isinstance(value, int)
            assert value > 0
            constants.num_processes = value
        elif key == "graph_chunksize":
            assert isinstance(value, int)
            assert value > 0
            constants.graph_chunksize = value
        elif key == "vertex_chunksize":
            assert isinstance(value, int)
            assert value > 0
            constants.vertex_chunksize = value
        elif key == "vector_buffer_size":
            assert isinstance(value, int)
            assert value > 0
            constants.vector_buffer_size = value
        else:
            raise ValueError(f'Invalid key {key} in config["general"]')

    for key, value in config["dataset"].items():
        if key == "dataset_str":
            assert isinstance(value, str)
            if value == 'ogbg-molhiv':
                dataset_str = value
            elif value == 'ogbg-ppa':
                dataset_str = value
            elif value == 'CSL':
                dataset_str = value
            elif value.endswith('-Prox'):
                h = int(value[0])
                assert isinstance(h, int) and h in [1,3,5,8,10]
                assert value == f'{h}-Prox'
                dataset_str = value
            else:
                raise ValueError(f'Invalid dataset_str: {value}')
        elif key == "k":
            if len(value) > 0:
                assert [x > 0 for x in value]
                k = value
        elif key == "r":
            if len(value) > 0:
                assert [x > 0 for x in value]
                if s is not None and len(s) > 0 and len(s) == len(value):
                    for idx in range(len(s)):
                        assert s[idx] >= value[idx]
                r = value
        elif key == "s":
            if len(value) > 0:
                assert [x > 0 for x in value]
                if r is not None and len(r) > 0 and len(r) == len(value):
                    for idx in range(len(r)):
                        assert r[idx] <= value[idx]
                s = value
        elif key == "gen_vertex_sp_features":
            assert isinstance(value, bool)
            gen_vertex_sp_features = value
        elif key == "re_gen_properties":
            assert isinstance(value, bool)
            re_gen_properties = value
        else:
            raise ValueError(f'Invalid key {key} in config["dataset"]')
        
    assert dataset_str is not None
    assert (k is not None and len(k) > 0) or (r is not None and s is not None and len(r) > 0) or gen_vertex_sp_features
    if r is not None:
        assert s is not None
    if s is not None:
        assert r is not None

    util.initialize_random_seeds(constants.SEED)

    try:
        if dataset_str == 'ogbg-molhiv':
            run_molhiv(k_vals = k, r_vals = r, s_vals = s, gen_vertex_sp_features = gen_vertex_sp_features, root_path = root_path, use_editmask = False, re_gen_properties = re_gen_properties)
        elif dataset_str == 'ogbg-ppa':
            run_ppa(k_vals = k, r_vals = r, s_vals = s, gen_vertex_sp_features = gen_vertex_sp_features, root_path = root_path, use_editmask = False, re_gen_properties = re_gen_properties)
        elif dataset_str == 'CSL':
            run_csl(k_vals = k, r_vals = r, s_vals = s, gen_vertex_sp_features = gen_vertex_sp_features, root_path = root_path, use_editmask = False, re_gen_properties = re_gen_properties)
        elif dataset_str.endswith('-Prox'):
            run_proximity(h_vals = [h], k_vals = k, r_vals = r, s_vals = s, gen_vertex_sp_features = gen_vertex_sp_features, root_path = root_path, use_editmask = False, re_gen_properties = re_gen_properties)
        else:
            raise ValueError('datasetstr')
    except Exception as e:
        print(repr(e))

def shorten_experiment_res_file(root_path: str, filename: str):
    data = util.read_metadata_file(path = osp.join(root_path, f'{filename}.json'))

    # Remove single epoch/rerun/fold data for classic gnns
    idx = 0
    while str(idx) in data["classic_gnn_hyperparameter_opt"]["experiment_idx"]:
        del data["classic_gnn_hyperparameter_opt"]["experiment_idx"][str(idx)]["splits"]
        idx += 1

    # Remove single epoch/rerun/fold data for clustering opt
    idx = 0
    while str(idx) in data["clustering_hyperparameter_opt"]["experiment_idx"]:
        split_idx = 0
        while str(split_idx) in data["clustering_hyperparameter_opt"]["experiment_idx"][str(idx)]["splits"]:
            del data["clustering_hyperparameter_opt"]["experiment_idx"][str(idx)]["splits"][str(split_idx)]["rerun"]
            split_idx += 1
        idx += 1

    # Remove single epoch/rerun/fold data for enhanced gnns
    idx = 0
    while str(idx) in data["enhanced_gnn_hyperparameter_opt"]["experiment_idx"]:
        del data["enhanced_gnn_hyperparameter_opt"]["experiment_idx"][str(idx)]["splits"]
        idx += 1

    util.write_metadata_file(path = root_path, filename = f'{filename}_short.json', data = data)

if __name__ == '__main__':
    # test gnn util

    mode = 1

    root_path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir)

    configs_path = osp.join(root_path, 'experiments', 'configs')

    if mode == 0:
        gen_experiment_config_file(root_path = root_path)
        gen_feature_gen_config_file(root_path = root_path)
    elif mode == 1:
        dirlist = os.listdir(path = configs_path)

        # We need to backup the experiment configs since we need to execute feature generations first
        experiment_configs = []

        # Required for pytorch version >= 2.6.0 since torch.load weights_only default value was changed from 'False' to 'True'
        torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

        for path in dirlist:
            if path.endswith('.json'):
                try:
                    config = util.read_metadata_file(path = osp.join(configs_path, path))
                    if "type" in config:
                        if config["type"] == "experiment":
                            # Run experiment
                            experiment_configs.append(config)
                        elif config["type"] == "feature_gen":
                            # Generate features
                            run_feature_gen(config = config, root_path = root_path)
                except Exception as e:
                    print(repr(e))
            
        # Run all experiments
        for idx, experiment_config in enumerate(experiment_configs):
            try:
                run_experiment(config = experiment_config, root_path = root_path, experiment_idx = idx)
            except Exception as e:
                print(repr(e))

    elif mode == 2:
        path = osp.join(root_path, 'experiments', 'results')
        filename = 'experiment_1_vertex_sp_gin'

        shorten_experiment_res_file(root_path = path, filename = filename)