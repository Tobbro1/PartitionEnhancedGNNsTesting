import os.path as osp
import numpy as np
import os
from typing import Dict, List
from pathlib import Path

import util
import constants
import argparse

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
    config["mode"] = "---   'classical', 'clustering', 'enhanced' or 'full' depending on whether a full model experiment or only a single step should be executed   ---"
    config["title"] = "---   filename of the result (without extension)   ---"
    config["prev_result_path"] = "---   path to a previous result if not classical or full experiment   ---"
    config["general"] = {}
    config["general"]["seed"] = constants.SEED
    config["general"]["num_workers"] = constants.num_workers
    config["general"]["num_reruns"] = constants.num_reruns
    config["general"]["max_patience"] = constants.max_patience
    config["general"]["use_batch_norm"] = constants.use_batch_norm
    config["general"]["num_k_fold"] = constants.num_k_fold
    config["general"]["k_fold_test_ratio"] = constants.k_fold_test_ratio
    config["general"]["mbk_batch_size"] = constants.mbk_batch_size
    config["general"]["mbk_num_init"] = constants.mbk_n_init
    config["general"]["mbk_max_no_improvement"] = constants.mbk_max_no_improvement
    config["general"]["mbk_max_iter"] = constants.mbk_max_iter
    config["dataset"] = {}
    config["dataset"]["dataset_str"] = "---   'ogbg-molhiv', 'ogbg-ppa', 'CSL' or 'h-Prox' with h = 1,3,5,8,10   ---"
    config["dataset"]["base_model"] = "---   'gin' or 'gcn'   ---"
    config["dataset"]["feature_type"] = "---   'sp' or 'lo', only utilised for k-disks or r-s-rings   ---"
    config["dataset"]["lo_feature_idx"] = "---   Index of the Lovasz features that should be utilised, use '0' if only one feature has been generated. Ignored if feature_type is not 'lo'   ---"
    config["dataset"]["k"] = ["---   List of k values for k-disks that should be evaluated   ---"]
    config["dataset"]["r"] = ["---   List of r values for r-s-rings that should be evaluated. NOTE: r[idx]-s[idx]-rings will be evaluated   ---"]
    config["dataset"]["s"] = ["---   List of s values for r-s-rings that should be evaluated. NOTE: r[idx]-s[idx]-rings will be evaluated   ---"]
    config["dataset"]["is_vertex_sp_feature"] = False
    config["dataset"]["normalize_vertex_features"] = False
    config["hyperparameters"] = {}
    config["hyperparameters"]["num_clusters"] = ["---   List of the numbers of clusters that will be evaluated   ---"]
    config["hyperparameters"]["pca_dims"] = ["---   List of the numbers of pca dimensions that will be evaluated. NOTE: values smaller than 1 mean no dimensionality reduction is performed   ---"]
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
    pca_dims = None
    min_cluster_sizes = None
    num_layers = None
    hidden_channels = None
    batch_sizes = None
    num_epochs = None
    lrs = None
    normalize = False
    dataset_str = ''
    base_model = ''
    exp_mode = -1 # 0 -> classical, 1 -> clustering, 2 -> enhanced, 3 -> full
    is_lovasz_feature = False
    lo_idx_str = ""

    prev_result_path = None

    # config["general"]["mbk_batch_size"] = constants.mbk_batch_size
    # config["general"]["mbk_num_init"] = constants.mbk_n_init
    # config["general"]["mbk_max_no_improvement"] = constants.mbk_max_no_improvement
    # config["general"]["mbk_max_iter"] = constants.mbk_max_iter

    if "mode" in config:
        val = config["mode"]
        assert isinstance(val, str)
        if val == "classical":
            exp_mode = 0
        elif val == "clustering":
            exp_mode = 1
            assert "prev_result_path" in config
            assert isinstance(config["prev_result_path"], str)
            prev_result_path = config["prev_result_path"]
        elif val == "enhanced":
            exp_mode = 2
            assert "prev_result_path" in config
            assert isinstance(config["prev_result_path"], str)
            prev_result_path = config["prev_result_path"]
        elif val == "full":
            exp_mode = 3
        else:
            raise ValueError('Invalid mode specified in config')

    # Parse config
    for key, value in config["general"].items():
        if key == "seed":
            assert isinstance(value, int)
            assert value >= 0
            constants.SEED = value
        elif key == "num_workers":
            assert isinstance(value, int)
            assert value >= 0
            constants.num_workers = value
        elif key == "num_reruns":
            assert isinstance(value, int)
            assert value > 0
            constants.num_reruns = value
        elif key == "max_patience":
            assert isinstance(value, int)
            assert value > 0
            constants.max_patience = value
        elif key == "use_batch_norm":
            assert isinstance(value, bool)
            constants.use_batch_norm = value
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
        elif key == "feature_type":
            assert isinstance(value, str)
            if value == "sp":
                is_lovasz_feature = False
            elif value == "lo":
                is_lovasz_feature = True
        elif key == "lo_feature_idx":
            assert isinstance(value, int)
            if value > 0:
                lo_idx_str = f"_{value}"
            else:
                lo_idx_str = ""
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
        elif key == "pca_dims":
            assert len(value) > 0
            assert [x > 0 for x in value]
            pca_dims = value
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
        manager.setup_experiments(dataset_str = dataset_str, base_model = base_model, is_lovasz_feature = is_lovasz_feature, k = k, r = r, s = s, is_vertex_sp_features = is_vertex_sp_features, num_clusters = num_clusters,
                                    pca_dims = pca_dims, min_cluster_sizes = min_cluster_sizes, num_layers = num_layers, hidden_channels = hidden_channels, batch_sizes = batch_sizes,
                                    num_epochs = num_epochs, lrs = lrs, normalize_features = normalize, exp_mode = exp_mode, max_patience = constants.max_patience, lo_idx_str = lo_idx_str,
                                    prev_result_path = prev_result_path)
        
        manager.run_experiments()

        if "title" in config and isinstance(config["title"], str) and len(config["title"]) > 0:
            filename = f'{config["title"]}.json'
        else:
            filename = f'experiment_{experiment_idx}_result.json'

        result_data = manager.get_metadata()
        result_data["result_path"] = osp.join(path, filename)

        util.write_metadata_file(data = result_data, path = path, filename = filename)

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
    config["general"]["num_lo_gen"] = constants.num_lo_gens
    config["dataset"] = {}
    config["dataset"]["dataset_str"] = "---   'ogbg-molhiv', 'ogbg-ppa', 'CSL' or 'h-Prox' with h = 1,3,5,8,10   ---"
    config["dataset"]["sp"] = {}
    config["dataset"]["sp"]["k"] = ["---   List of k values for k-disks that should be evaluated   ---"]
    config["dataset"]["sp"]["r"] = ["---   List of r values for r-s-rings that should be evaluated. NOTE: r[idx]-s[idx]-rings will be evaluated   ---"]
    config["dataset"]["sp"]["s"] = ["---   List of s values for r-s-rings that should be evaluated. NOTE: r[idx]-s[idx]-rings will be evaluated   ---"]
    config["dataset"]["sp"]["gen_vertex_sp_features"] = False
    config["dataset"]["lo"] = {}
    config["dataset"]["lo"]["size_smallest_subgraph"] = "---   Size of the smallest subgraph that should be considered when computing the Lovasz feature   ---"
    config["dataset"]["lo"]["size_largest_subgraph"] = "---   Size of the largest subgraph that should be considered when computing the Lovasz feature   ---"
    config["dataset"]["lo"]["num_subgraph_samples"] = "---   Number of subgraphs that should be considered when computing the Lovasz feature   ---"
    config["dataset"]["lo"]["k"] = ["---   List of k values for k-disks that should be evaluated   ---"]
    config["dataset"]["lo"]["r"] = ["---   List of r values for r-s-rings that should be evaluated. NOTE: r[idx]-s[idx]-rings will be evaluated   ---"]
    config["dataset"]["lo"]["s"] = ["---   List of s values for r-s-rings that should be evaluated. NOTE: r[idx]-s[idx]-rings will be evaluated   ---"]
    config["dataset"]["re_gen_properties"] = True

    util.write_metadata_file(path = path, filename = filename, data = config)

def run_feature_gen(config: Dict, root_path: str) -> None:


    # Parse the config and sanitize the input
    assert config["type"] == "feature_gen"
    assert "general" in config
    assert "dataset" in config

    h = None
    sp_k = None
    sp_r = None
    sp_s = None
    lo_k = None
    lo_r = None
    lo_s = None
    lo_graph_sizes_range = [-1,-1]
    lo_num_samples = None
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
        elif key == "num_lo_gen":
            assert isinstance(value, int)
            assert value > 0
            constants.num_lo_gens = value
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
        elif key == "sp":
            for k, v in value.items():
                if k == "k":
                    if len(v) > 0:
                        assert [x > 0 for x in v]
                        sp_k = v
                elif k == "r":
                    if len(v) > 0:
                        assert [x > 0 for x in v]
                        if sp_s is not None and len(sp_s) > 0 and len(sp_s) == len(v):
                            for idx in range(len(sp_s)):
                                assert sp_s[idx] >= v[idx]
                        sp_r = v
                elif k == "s":
                    if len(v) > 0:
                        assert [x > 0 for x in v]
                        if sp_r is not None and len(sp_r) > 0 and len(sp_r) == len(v):
                            for idx in range(len(sp_r)):
                                assert sp_r[idx] <= v[idx]
                        sp_s = v
                elif k == "gen_vertex_sp_features":
                    assert isinstance(v, bool)
                    gen_vertex_sp_features = v
        elif key == "lo":
            for k, v in value.items():
                if k == "size_smallest_subgraph":
                    assert isinstance(v, int)
                    assert v > 0
                    if lo_graph_sizes_range[1] != -1:
                        assert v <= lo_graph_sizes_range[1]
                    lo_graph_sizes_range[0] = v
                elif k == "size_largest_subgraph":
                    assert isinstance(v, int)
                    assert v > 0
                    if lo_graph_sizes_range[0] != -1:
                        assert v >= lo_graph_sizes_range[0]
                    lo_graph_sizes_range[1] = v
                elif k == "num_subgraph_samples":
                    assert isinstance(v, int)
                    assert v > 0
                    lo_num_samples = v
                if k == "k":
                    if len(v) > 0:
                        assert [x > 0 for x in v]
                        lo_k = v
                elif k == "r":
                    if len(v) > 0:
                        assert [x > 0 for x in v]
                        if lo_s is not None and len(lo_s) > 0 and len(lo_s) == len(v):
                            for idx in range(len(lo_s)):
                                assert lo_s[idx] >= v[idx]
                        lo_r = v
                elif k == "s":
                    if len(v) > 0:
                        assert [x > 0 for x in v]
                        if lo_r is not None and len(lo_r) > 0 and len(lo_r) == len(v):
                            for idx in range(len(lo_r)):
                                assert lo_r[idx] <= v[idx]
                        lo_s = v
        elif key == "re_gen_properties":
            assert isinstance(value, bool)
            re_gen_properties = value
        else:
            raise ValueError(f'Invalid key {key} in config["dataset"]')
    
    lo_graph_sizes_range = tuple(lo_graph_sizes_range)

    assert dataset_str is not None
    assert (sp_k is not None and len(sp_k) > 0) or (sp_r is not None and sp_s is not None and len(sp_r) > 0) or (lo_k is not None and len(lo_k) > 0) or (lo_r is not None and lo_s is not None and len(lo_r) > 0) or gen_vertex_sp_features
    if sp_r is not None:
        assert sp_s is not None
    if sp_s is not None:
        assert sp_r is not None
    if lo_r is not None:
        assert lo_s is not None
    if lo_s is not None:
        assert lo_r is not None

    util.initialize_random_seeds(constants.SEED)

    if dataset_str == 'ogbg-molhiv':
        run_molhiv(sp_k_vals = sp_k, sp_r_vals = sp_r, sp_s_vals = sp_s, lo_k_vals = lo_k, lo_r_vals = lo_r, lo_s_vals = lo_s, lo_graph_sizes_range = lo_graph_sizes_range, lo_num_samples = lo_num_samples, gen_vertex_sp_features = gen_vertex_sp_features, root_path = root_path, use_editmask = False, re_gen_properties = re_gen_properties)
    elif dataset_str == 'ogbg-ppa':
        run_ppa(sp_k_vals = sp_k, sp_r_vals = sp_r, sp_s_vals = sp_s, lo_k_vals = lo_k, lo_r_vals = lo_r, lo_s_vals = lo_s, lo_graph_sizes_range = lo_graph_sizes_range, lo_num_samples = lo_num_samples, gen_vertex_sp_features = gen_vertex_sp_features, root_path = root_path, use_editmask = False, re_gen_properties = re_gen_properties)
    elif dataset_str == 'CSL':
        run_csl(sp_k_vals = sp_k, sp_r_vals = sp_r, sp_s_vals = sp_s, lo_k_vals = lo_k, lo_r_vals = lo_r, lo_s_vals = lo_s, lo_graph_sizes_range = lo_graph_sizes_range, lo_num_samples = lo_num_samples, gen_vertex_sp_features = gen_vertex_sp_features, root_path = root_path, use_editmask = False, re_gen_properties = re_gen_properties)
    elif dataset_str.endswith('-Prox'):
        run_proximity(h_vals = [h], sp_k_vals = sp_k, sp_r_vals = sp_r, sp_s_vals = sp_s, lo_k_vals = lo_k, lo_r_vals = lo_r, lo_s_vals = lo_s, lo_graph_sizes_range = lo_graph_sizes_range, lo_num_samples = lo_num_samples, gen_vertex_sp_features = gen_vertex_sp_features, root_path = root_path, use_editmask = False, re_gen_properties = re_gen_properties)
    else:
        raise ValueError('datasetstr')


    # try:
    #     if dataset_str == 'ogbg-molhiv':
    #         run_molhiv(sp_k_vals = sp_k, sp_r_vals = sp_r, sp_s_vals = sp_s, lo_k_vals = lo_k, lo_r_vals = lo_r, lo_s_vals = lo_s, lo_graph_sizes_range = lo_graph_sizes_range, lo_num_samples = lo_num_samples, gen_vertex_sp_features = gen_vertex_sp_features, root_path = root_path, use_editmask = False, re_gen_properties = re_gen_properties)
    #     elif dataset_str == 'ogbg-ppa':
    #         run_ppa(sp_k_vals = sp_k, sp_r_vals = sp_r, sp_s_vals = sp_s, lo_k_vals = lo_k, lo_r_vals = lo_r, lo_s_vals = lo_s, lo_graph_sizes_range = lo_graph_sizes_range, lo_num_samples = lo_num_samples, gen_vertex_sp_features = gen_vertex_sp_features, root_path = root_path, use_editmask = False, re_gen_properties = re_gen_properties)
    #     elif dataset_str == 'CSL':
    #         run_csl(sp_k_vals = sp_k, sp_r_vals = sp_r, sp_s_vals = sp_s, lo_k_vals = lo_k, lo_r_vals = lo_r, lo_s_vals = lo_s, lo_graph_sizes_range = lo_graph_sizes_range, lo_num_samples = lo_num_samples, gen_vertex_sp_features = gen_vertex_sp_features, root_path = root_path, use_editmask = False, re_gen_properties = re_gen_properties)
    #     elif dataset_str.endswith('-Prox'):
    #         run_proximity(h_vals = [h], sp_k_vals = sp_k, sp_r_vals = sp_r, sp_s_vals = sp_s, lo_k_vals = lo_k, lo_r_vals = lo_r, lo_s_vals = lo_s, lo_graph_sizes_range = lo_graph_sizes_range, lo_num_samples = lo_num_samples, gen_vertex_sp_features = gen_vertex_sp_features, root_path = root_path, use_editmask = False, re_gen_properties = re_gen_properties)
    #     else:
    #         raise ValueError('datasetstr')
    # except Exception as e:
    #     print(repr(e))

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

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-g', '--generate', help = 'generate example files for configs', action = 'store_true')
    group.add_argument('-f', '--file', nargs = '?', help = 'specify path to a config file that will be executed', type = Path)
    group.add_argument('-d', '--directory', nargs = '?', help = 'specify path to a directory which will be scanned for config files which will be executed consecutively', type = Path)
    parser.add_argument('-r', '--root_path', nargs = '?', help = 'specify the root directory, if not set will default to the parent directory of the main.py file', type = Path, default = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir))

    args = parser.parse_args()

    # Required for pytorch version >= 2.6.0 since torch.load weights_only default value was changed from 'False' to 'True'
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

    if args.generate:
        # generate config files
        gen_experiment_config_file(root_path = args.root_path)
        gen_feature_gen_config_file(root_path = args.root_path)
    elif args.file is not None:
        # execute the file
        config = util.read_metadata_file(path = args.file)
        if "type" in config:
            if config["type"] == "experiment":
                # Run experiment
                run_experiment(config, root_path = args.root_path, experiment_idx = 0)
            elif config["type"] == "feature_gen":
                # Generate features
                run_feature_gen(config = config, root_path = args.root_path)
        else:
            raise ValueError('Invalid config - no type property')
    elif args.directory is not None:
        dirlist = os.listdir(path = args.directory)

        # We need to backup the experiment configs since we need to execute feature generations first
        experiment_configs = []

        for path in dirlist:
            if path.endswith('.json'):
                try:
                    config = util.read_metadata_file(path = osp.join(args.directory, path))
                    if "type" in config:
                        if config["type"] == "experiment":
                            # Run experiment
                            experiment_configs.append(config)
                        elif config["type"] == "feature_gen":
                            # Generate features
                            run_feature_gen(config = config, root_path = args.root_path)
                except Exception as e:
                    print(repr(e))
            
        # Run all experiments
        for idx, experiment_config in enumerate(experiment_configs):
            try:
                run_experiment(config = experiment_config, root_path = args.root_path, experiment_idx = idx)
            except Exception as e:
                print(repr(e))

    # elif mode == 2:
    #     path = osp.join(root_path, 'experiments', 'results')
    #     filename = 'experiment_1_vertex_sp_gin'

    #     shorten_experiment_res_file(root_path = path, filename = filename)