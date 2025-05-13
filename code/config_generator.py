# Script used to generate some configs, not important and does not hold funtionality

import os.path as osp
import util
import constants

output_root_folder = "E:\\CopyToOtherPC\\experiments\\configs\\COLLAB" # "E:\\CopyToOtherPC\\experiments\\configs\\3-Prox\\augmented_gnn"
prev_res_folder = "/home/tobias/bachelorthesis/PartitionEnhancedGNNsTesting/experiments/results" # "E:\\workspace\\PartitionEnhancedGNNsTesting\\experiments\\results" #  #  # 
stage = 2 # 0 -> only classical, 1 -> only clustering, 2 -> only enhanced/augmented, 3 -> all stages

k_vals = [2]
r_vals = [1, 2]
s_vals = [2, 2]
vertex_sp = True
num_reruns = 3
max_patience = 5

num_clusters = [2, 3, 5, 10, 20]
pca_dims = [0, 10, 30]
min_cluster_size = [0]
num_layers = [3, 5]
num_hidden_channels = [64, 128, 256]
num_batch_sizes = [32]
num_epochs = [100]
lrs = [0.01]

dataset = "COLLAB"
models = ["gin", "gcn"]
# create both models
# create with and without batch norm for final enhanced gnn version
# create normalized and non-normalized vertex features
is_linux = True
gen_enhanced_gnn = False
gen_augmented_gnn = True

feature_type = "sp"
lo_feature_index = 0
num_gpnn_layers = [2]
num_gpnn_channels = [32]

num_workers = constants.num_workers
num_k_fold = constants.num_k_fold
k_fold_test_ratio = constants.k_fold_test_ratio
mbk_batch_size = constants.mbk_batch_size
mbk_num_init = constants.mbk_n_init
mbk_max_no_improvement = constants.mbk_max_no_improvement
mbk_max_iter = constants.mbk_max_iter


# config = {}
# config["type"] = "experiment"
# config["mode"] = "---   'classical', 'clustering', 'enhanced' or 'full' depending on whether a full model experiment or only a single step should be executed   ---"
# config["title"] = "---   filename of the result (without extension)   ---"
# config["prev_result_path"] = "---   path to a previous result if not classical or full experiment   ---"
# config["general"] = {}
# config["general"]["seed"] = constants.SEED
# config["general"]["num_workers"] = constants.num_workers
# config["general"]["num_reruns"] = constants.num_reruns
# config["general"]["max_patience"] = constants.max_patience
# config["general"]["use_batch_norm"] = constants.use_batch_norm
# config["general"]["num_k_fold"] = constants.num_k_fold
# config["general"]["k_fold_test_ratio"] = constants.k_fold_test_ratio
# config["general"]["mbk_batch_size"] = constants.mbk_batch_size
# config["general"]["mbk_num_init"] = constants.mbk_n_init
# config["general"]["mbk_max_no_improvement"] = constants.mbk_max_no_improvement
# config["general"]["mbk_max_iter"] = constants.mbk_max_iter
# config["dataset"] = {}
# config["dataset"]["dataset_str"] = "---   'NCI1', 'ENZYMES', 'PROTEINS', 'DD', 'ogbg-molhiv', 'ogbg-ppa', 'CSL' or 'h-Prox' with h = 1,3,5,8,10   ---"
# config["dataset"]["base_model"] = "---   'gin' or 'gcn'   ---"
# config["dataset"]["use_gpnn"] = False
# config["dataset"]["use_augmented_gnn"] = False
# config["dataset"]["feature_type"] = "---   'sp' or 'lo', only utilised for k-disks or r-s-rings   ---"
# config["dataset"]["lo_feature_idx"] = "---   Index of the Lovasz features that should be utilised, use '0' if only one feature has been generated. Ignored if feature_type is not 'lo'   ---"
# config["dataset"]["k"] = ["---   List of k values for k-disks that should be evaluated   ---"]
# config["dataset"]["r"] = ["---   List of r values for r-s-rings that should be evaluated. NOTE: r[idx]-s[idx]-rings will be evaluated   ---"]
# config["dataset"]["s"] = ["---   List of s values for r-s-rings that should be evaluated. NOTE: r[idx]-s[idx]-rings will be evaluated   ---"]
# config["dataset"]["is_vertex_sp_feature"] = False
# config["dataset"]["normalize_vertex_features"] = False
# config["hyperparameters"] = {}
# config["hyperparameters"]["num_clusters"] = ["---   List of the numbers of clusters that will be evaluated   ---"]
# config["hyperparameters"]["pca_dims"] = ["---   List of the numbers of pca dimensions that will be evaluated. NOTE: values smaller than 1 mean no dimensionality reduction is performed   ---"]
# config["hyperparameters"]["min_cluster_size"] = ["---   List of the minimum sizes of clusters that will be evaluated   ---"]
# config["hyperparameters"]["num_layers"] = ["---   List of the number of layers of gnns that will be evaluated   ---"]
# config["hyperparameters"]["num_hidden_channels"] = ["---   List of the number hidden dimensions of gnns that will be evaluated   ---"]
# config["hyperparameters"]["num_batch_sizes"] = ["---   Defines the batch sizes of gnns while training   ---"]
# config["hyperparameters"]["num_epochs"] = ["---   List of the number of epochs while training gnns that will be evaluated   ---"]
# config["hyperparameters"]["lrs"] = ["---   List of the learning rates that will be evaluated   ---"]
# config["hyperparameters"]["num_gpnn_layer"] = ["---   List of the number of layers of gpnns that will be evaluated   ---"]
# config["hyperparameters"]["gpnn_channels"] = ["---   List of the gpnn feature dimensions that will be evaluated   ---"]

# We split into different configs: feature type?, enhanced/augmented gnn?, model?, batchnorm?, for enhanced: num_layers; for clustering: num_clusters, normalize?
def gen_classical_config():
    batch_norms = [False, True]
    
    prev_result_path = ""

    for model in models:
        for bn in batch_norms:
            if bn:
                title = f"{dataset}_classical_{model}_bn"
            else:
                title = f"{dataset}_classical_{model}"

            filename = f"{title}.json"

            if bn:
                path = osp.join(output_root_folder, "classical_gnn", model, "bn")
            else:
                path = osp.join(output_root_folder, "classical_gnn", model, "normal")

            # gen config
            config = {}
            config["type"] = "experiment"
            config["mode"] = "classical"
            config["title"] = title
            config["prev_result_path"] = prev_result_path
            config["general"] = {}
            config["general"]["seed"] = constants.SEED
            config["general"]["num_workers"] = constants.num_workers
            config["general"]["num_reruns"] = num_reruns
            config["general"]["max_patience"] = max_patience
            config["general"]["use_batch_norm"] = bn
            config["general"]["num_k_fold"] = num_k_fold
            config["general"]["k_fold_test_ratio"] = k_fold_test_ratio
            config["general"]["mbk_batch_size"] = mbk_batch_size
            config["general"]["mbk_num_init"] = mbk_num_init
            config["general"]["mbk_max_no_improvement"] = mbk_max_no_improvement
            config["general"]["mbk_max_iter"] = mbk_max_iter

            config["dataset"] = {}
            config["dataset"]["dataset_str"] = dataset
            config["dataset"]["base_model"] = model
            config["dataset"]["use_gpnn"] = False
            config["dataset"]["use_augmented_gnn"] = False
            config["dataset"]["feature_type"] = feature_type
            config["dataset"]["lo_feature_idx"] = lo_feature_index
            config["dataset"]["k"] = k_vals
            config["dataset"]["r"] = r_vals
            config["dataset"]["s"] = s_vals
            config["dataset"]["is_vertex_sp_feature"] = False
            config["dataset"]["normalize_vertex_features"] = False
            config["hyperparameters"] = {}
            config["hyperparameters"]["num_clusters"] = num_clusters
            config["hyperparameters"]["pca_dims"] = pca_dims
            config["hyperparameters"]["min_cluster_size"] = min_cluster_size
            config["hyperparameters"]["num_layers"] = num_layers
            config["hyperparameters"]["num_hidden_channels"] = num_hidden_channels
            config["hyperparameters"]["num_batch_sizes"] = num_batch_sizes
            config["hyperparameters"]["num_epochs"] = num_epochs
            config["hyperparameters"]["lrs"] = lrs
            config["hyperparameters"]["num_gpnn_layer"] = num_gpnn_layers
            config["hyperparameters"]["gpnn_channels"] = num_gpnn_channels

            util.write_metadata_file(path = path, filename = filename, data = config)

def gen_clustering_config():
    normalize_features = [False, True]
    
    for model in models:
        for norm in normalize_features:
            for n_cluster in num_clusters:
                for k in k_vals:
                    if gen_enhanced_gnn:
                        if norm:
                            title = f"{dataset}_{k}-disk_norm_clustering_{n_cluster}c_enhanced_{model}"
                        else:
                            title = f"{dataset}_{k}-disk_clustering_{n_cluster}c_enhanced_{model}"

                        filename = f"{title}.json"

                        path = osp.join(output_root_folder, "clustering_enhanced_gnn", f"{k}-disk", f"{n_cluster}-cluster", model)

                        if is_linux:
                            prev_result_path = f"{prev_res_folder}/{f"{dataset}_classical_{model}"}.json"
                        else:
                            prev_result_path = f"{prev_res_folder}\\{f"{dataset}_classical_{model}"}.json"

                        # gen config
                        config = {}
                        config["type"] = "experiment"
                        config["mode"] = "clustering"
                        config["title"] = title
                        config["prev_result_path"] = prev_result_path
                        config["general"] = {}
                        config["general"]["seed"] = constants.SEED
                        config["general"]["num_workers"] = constants.num_workers
                        config["general"]["num_reruns"] = num_reruns
                        config["general"]["max_patience"] = max_patience
                        config["general"]["use_batch_norm"] = False
                        config["general"]["num_k_fold"] = num_k_fold
                        config["general"]["k_fold_test_ratio"] = k_fold_test_ratio
                        config["general"]["mbk_batch_size"] = mbk_batch_size
                        config["general"]["mbk_num_init"] = mbk_num_init
                        config["general"]["mbk_max_no_improvement"] = mbk_max_no_improvement
                        config["general"]["mbk_max_iter"] = mbk_max_iter

                        config["dataset"] = {}
                        config["dataset"]["dataset_str"] = dataset
                        config["dataset"]["base_model"] = model
                        config["dataset"]["use_gpnn"] = False
                        config["dataset"]["use_augmented_gnn"] = False
                        config["dataset"]["feature_type"] = feature_type
                        config["dataset"]["lo_feature_idx"] = lo_feature_index
                        config["dataset"]["k"] = [k]
                        config["dataset"]["r"] = []
                        config["dataset"]["s"] = []
                        config["dataset"]["is_vertex_sp_feature"] = False
                        config["dataset"]["normalize_vertex_features"] = norm
                        config["hyperparameters"] = {}
                        config["hyperparameters"]["num_clusters"] = [n_cluster]
                        config["hyperparameters"]["pca_dims"] = pca_dims
                        config["hyperparameters"]["min_cluster_size"] = min_cluster_size
                        config["hyperparameters"]["num_layers"] = num_layers
                        config["hyperparameters"]["num_hidden_channels"] = num_hidden_channels
                        config["hyperparameters"]["num_batch_sizes"] = num_batch_sizes
                        config["hyperparameters"]["num_epochs"] = num_epochs
                        config["hyperparameters"]["lrs"] = lrs
                        config["hyperparameters"]["num_gpnn_layer"] = num_gpnn_layers
                        config["hyperparameters"]["gpnn_channels"] = num_gpnn_channels

                        util.write_metadata_file(path = path, filename = filename, data = config)

                    if gen_augmented_gnn:
                        if norm:
                            title = f"{dataset}_{k}-disk_norm_clustering_{n_cluster}c_augmented_{model}"
                        else:
                            title = f"{dataset}_{k}-disk_clustering_{n_cluster}c_augmented_{model}"

                        filename = f"{title}.json"

                        path = osp.join(output_root_folder, "clustering_augmented_gnn", f"{k}-disk", f"{n_cluster}-cluster", model)

                        if is_linux:
                            prev_result_path = f"{prev_res_folder}/{f"{dataset}_classical_{model}"}.json"
                        else:
                            prev_result_path = f"{prev_res_folder}\\{f"{dataset}_classical_{model}"}.json"

                        # gen config
                        config = {}
                        config["type"] = "experiment"
                        config["mode"] = "clustering"
                        config["title"] = title
                        config["prev_result_path"] = prev_result_path
                        config["general"] = {}
                        config["general"]["seed"] = constants.SEED
                        config["general"]["num_workers"] = constants.num_workers
                        config["general"]["num_reruns"] = num_reruns
                        config["general"]["max_patience"] = max_patience
                        config["general"]["use_batch_norm"] = False
                        config["general"]["num_k_fold"] = num_k_fold
                        config["general"]["k_fold_test_ratio"] = k_fold_test_ratio
                        config["general"]["mbk_batch_size"] = mbk_batch_size
                        config["general"]["mbk_num_init"] = mbk_num_init
                        config["general"]["mbk_max_no_improvement"] = mbk_max_no_improvement
                        config["general"]["mbk_max_iter"] = mbk_max_iter

                        config["dataset"] = {}
                        config["dataset"]["dataset_str"] = dataset
                        config["dataset"]["base_model"] = model
                        config["dataset"]["use_gpnn"] = False
                        config["dataset"]["use_augmented_gnn"] = True
                        config["dataset"]["feature_type"] = feature_type
                        config["dataset"]["lo_feature_idx"] = lo_feature_index
                        config["dataset"]["k"] = [k]
                        config["dataset"]["r"] = []
                        config["dataset"]["s"] = []
                        config["dataset"]["is_vertex_sp_feature"] = False
                        config["dataset"]["normalize_vertex_features"] = norm
                        config["hyperparameters"] = {}
                        config["hyperparameters"]["num_clusters"] = [n_cluster]
                        config["hyperparameters"]["pca_dims"] = pca_dims
                        config["hyperparameters"]["min_cluster_size"] = min_cluster_size
                        config["hyperparameters"]["num_layers"] = num_layers
                        config["hyperparameters"]["num_hidden_channels"] = num_hidden_channels
                        config["hyperparameters"]["num_batch_sizes"] = num_batch_sizes
                        config["hyperparameters"]["num_epochs"] = num_epochs
                        config["hyperparameters"]["lrs"] = lrs
                        config["hyperparameters"]["num_gpnn_layer"] = num_gpnn_layers
                        config["hyperparameters"]["gpnn_channels"] = num_gpnn_channels

                        util.write_metadata_file(path = path, filename = filename, data = config)

                for idx, _ in enumerate(r_vals):
                    if gen_enhanced_gnn:
                        if norm:
                            title = f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_norm_clustering_{n_cluster}c_enhanced_{model}"
                        else:
                            title = f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_clustering_{n_cluster}c_enhanced_{model}"

                        filename = f"{title}.json"

                        path = osp.join(output_root_folder, "clustering_enhanced_gnn", f"{r_vals[idx]}-{s_vals[idx]}-ring", f"{n_cluster}-cluster", model)

                        if is_linux:
                            prev_result_path = f"{prev_res_folder}/{f"{dataset}_classical_{model}"}.json"
                        else:
                            prev_result_path = f"{prev_res_folder}\\{f"{dataset}_classical_{model}"}.json"

                        # gen config
                        config = {}
                        config["type"] = "experiment"
                        config["mode"] = "clustering"
                        config["title"] = title
                        config["prev_result_path"] = prev_result_path
                        config["general"] = {}
                        config["general"]["seed"] = constants.SEED
                        config["general"]["num_workers"] = constants.num_workers
                        config["general"]["num_reruns"] = num_reruns
                        config["general"]["max_patience"] = max_patience
                        config["general"]["use_batch_norm"] = False
                        config["general"]["num_k_fold"] = num_k_fold
                        config["general"]["k_fold_test_ratio"] = k_fold_test_ratio
                        config["general"]["mbk_batch_size"] = mbk_batch_size
                        config["general"]["mbk_num_init"] = mbk_num_init
                        config["general"]["mbk_max_no_improvement"] = mbk_max_no_improvement
                        config["general"]["mbk_max_iter"] = mbk_max_iter

                        config["dataset"] = {}
                        config["dataset"]["dataset_str"] = dataset
                        config["dataset"]["base_model"] = model
                        config["dataset"]["use_gpnn"] = False
                        config["dataset"]["use_augmented_gnn"] = False
                        config["dataset"]["feature_type"] = feature_type
                        config["dataset"]["lo_feature_idx"] = lo_feature_index
                        config["dataset"]["k"] = []
                        config["dataset"]["r"] = [r_vals[idx]]
                        config["dataset"]["s"] = [s_vals[idx]]
                        config["dataset"]["is_vertex_sp_feature"] = False
                        config["dataset"]["normalize_vertex_features"] = norm
                        config["hyperparameters"] = {}
                        config["hyperparameters"]["num_clusters"] = [n_cluster]
                        config["hyperparameters"]["pca_dims"] = pca_dims
                        config["hyperparameters"]["min_cluster_size"] = min_cluster_size
                        config["hyperparameters"]["num_layers"] = num_layers
                        config["hyperparameters"]["num_hidden_channels"] = num_hidden_channels
                        config["hyperparameters"]["num_batch_sizes"] = num_batch_sizes
                        config["hyperparameters"]["num_epochs"] = num_epochs
                        config["hyperparameters"]["lrs"] = lrs
                        config["hyperparameters"]["num_gpnn_layer"] = num_gpnn_layers
                        config["hyperparameters"]["gpnn_channels"] = num_gpnn_channels

                        util.write_metadata_file(path = path, filename = filename, data = config)

                    if gen_augmented_gnn:
                        if norm:
                            title = f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_norm_clustering_{n_cluster}c_augmented_{model}"
                        else:
                            title = f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_clustering_{n_cluster}c_augmented_{model}"

                        filename = f"{title}.json"

                        path = osp.join(output_root_folder, "clustering_augmented_gnn", f"{r_vals[idx]}-{s_vals[idx]}-ring", f"{n_cluster}-cluster", model)

                        if is_linux:
                            prev_result_path = f"{prev_res_folder}/{f"{dataset}_classical_{model}"}.json"
                        else:
                            prev_result_path = f"{prev_res_folder}\\{f"{dataset}_classical_{model}"}.json"

                        # gen config
                        config = {}
                        config["type"] = "experiment"
                        config["mode"] = "clustering"
                        config["title"] = title
                        config["prev_result_path"] = prev_result_path
                        config["general"] = {}
                        config["general"]["seed"] = constants.SEED
                        config["general"]["num_workers"] = constants.num_workers
                        config["general"]["num_reruns"] = num_reruns
                        config["general"]["max_patience"] = max_patience
                        config["general"]["use_batch_norm"] = False
                        config["general"]["num_k_fold"] = num_k_fold
                        config["general"]["k_fold_test_ratio"] = k_fold_test_ratio
                        config["general"]["mbk_batch_size"] = mbk_batch_size
                        config["general"]["mbk_num_init"] = mbk_num_init
                        config["general"]["mbk_max_no_improvement"] = mbk_max_no_improvement
                        config["general"]["mbk_max_iter"] = mbk_max_iter

                        config["dataset"] = {}
                        config["dataset"]["dataset_str"] = dataset
                        config["dataset"]["base_model"] = model
                        config["dataset"]["use_gpnn"] = False
                        config["dataset"]["use_augmented_gnn"] = True
                        config["dataset"]["feature_type"] = feature_type
                        config["dataset"]["lo_feature_idx"] = lo_feature_index
                        config["dataset"]["k"] = []
                        config["dataset"]["r"] = [r_vals[idx]]
                        config["dataset"]["s"] = [s_vals[idx]]
                        config["dataset"]["is_vertex_sp_feature"] = False
                        config["dataset"]["normalize_vertex_features"] = norm
                        config["hyperparameters"] = {}
                        config["hyperparameters"]["num_clusters"] = [n_cluster]
                        config["hyperparameters"]["pca_dims"] = pca_dims
                        config["hyperparameters"]["min_cluster_size"] = min_cluster_size
                        config["hyperparameters"]["num_layers"] = num_layers
                        config["hyperparameters"]["num_hidden_channels"] = num_hidden_channels
                        config["hyperparameters"]["num_batch_sizes"] = num_batch_sizes
                        config["hyperparameters"]["num_epochs"] = num_epochs
                        config["hyperparameters"]["lrs"] = lrs
                        config["hyperparameters"]["num_gpnn_layer"] = num_gpnn_layers
                        config["hyperparameters"]["gpnn_channels"] = num_gpnn_channels

                        util.write_metadata_file(path = path, filename = filename, data = config)

                if vertex_sp:
                    if gen_enhanced_gnn:
                        if norm:
                            title = f"{dataset}_vsp_norm_clustering_{n_cluster}c_enhanced_{model}"
                        else:
                            title = f"{dataset}_vsp_clustering_{n_cluster}c_enhanced_{model}"

                        filename = f"{title}.json"

                        path = osp.join(output_root_folder, "clustering_enhanced_gnn", f"vsp", f"{n_cluster}-cluster", model)

                        if is_linux:
                            prev_result_path = f"{prev_res_folder}/{f"{dataset}_classical_{model}"}.json"
                        else:
                            prev_result_path = f"{prev_res_folder}\\{f"{dataset}_classical_{model}"}.json"

                        # gen config
                        config = {}
                        config["type"] = "experiment"
                        config["mode"] = "clustering"
                        config["title"] = title
                        config["prev_result_path"] = prev_result_path
                        config["general"] = {}
                        config["general"]["seed"] = constants.SEED
                        config["general"]["num_workers"] = constants.num_workers
                        config["general"]["num_reruns"] = num_reruns
                        config["general"]["max_patience"] = max_patience
                        config["general"]["use_batch_norm"] = False
                        config["general"]["num_k_fold"] = num_k_fold
                        config["general"]["k_fold_test_ratio"] = k_fold_test_ratio
                        config["general"]["mbk_batch_size"] = mbk_batch_size
                        config["general"]["mbk_num_init"] = mbk_num_init
                        config["general"]["mbk_max_no_improvement"] = mbk_max_no_improvement
                        config["general"]["mbk_max_iter"] = mbk_max_iter

                        config["dataset"] = {}
                        config["dataset"]["dataset_str"] = dataset
                        config["dataset"]["base_model"] = model
                        config["dataset"]["use_gpnn"] = False
                        config["dataset"]["use_augmented_gnn"] = False
                        config["dataset"]["feature_type"] = feature_type
                        config["dataset"]["lo_feature_idx"] = lo_feature_index
                        config["dataset"]["k"] = []
                        config["dataset"]["r"] = []
                        config["dataset"]["s"] = []
                        config["dataset"]["is_vertex_sp_feature"] = True
                        config["dataset"]["normalize_vertex_features"] = norm
                        config["hyperparameters"] = {}
                        config["hyperparameters"]["num_clusters"] = [n_cluster]
                        config["hyperparameters"]["pca_dims"] = pca_dims
                        config["hyperparameters"]["min_cluster_size"] = min_cluster_size
                        config["hyperparameters"]["num_layers"] = num_layers
                        config["hyperparameters"]["num_hidden_channels"] = num_hidden_channels
                        config["hyperparameters"]["num_batch_sizes"] = num_batch_sizes
                        config["hyperparameters"]["num_epochs"] = num_epochs
                        config["hyperparameters"]["lrs"] = lrs
                        config["hyperparameters"]["num_gpnn_layer"] = num_gpnn_layers
                        config["hyperparameters"]["gpnn_channels"] = num_gpnn_channels

                        util.write_metadata_file(path = path, filename = filename, data = config)

                    if gen_augmented_gnn:
                        if norm:
                            title = f"{dataset}_vsp_norm_clustering_{n_cluster}c_augmented_{model}"
                        else:
                            title = f"{dataset}_vsp_clustering_{n_cluster}c_augmented_{model}"

                        filename = f"{title}.json"

                        path = osp.join(output_root_folder, "clustering_augmented_gnn", f"vsp", f"{n_cluster}-cluster", model)

                        if is_linux:
                            prev_result_path = f"{prev_res_folder}/{f"{dataset}_classical_{model}"}.json"
                        else:
                            prev_result_path = f"{prev_res_folder}\\{f"{dataset}_classical_{model}"}.json"

                        # gen config
                        config = {}
                        config["type"] = "experiment"
                        config["mode"] = "clustering"
                        config["title"] = title
                        config["prev_result_path"] = prev_result_path
                        config["general"] = {}
                        config["general"]["seed"] = constants.SEED
                        config["general"]["num_workers"] = constants.num_workers
                        config["general"]["num_reruns"] = num_reruns
                        config["general"]["max_patience"] = max_patience
                        config["general"]["use_batch_norm"] = False
                        config["general"]["num_k_fold"] = num_k_fold
                        config["general"]["k_fold_test_ratio"] = k_fold_test_ratio
                        config["general"]["mbk_batch_size"] = mbk_batch_size
                        config["general"]["mbk_num_init"] = mbk_num_init
                        config["general"]["mbk_max_no_improvement"] = mbk_max_no_improvement
                        config["general"]["mbk_max_iter"] = mbk_max_iter

                        config["dataset"] = {}
                        config["dataset"]["dataset_str"] = dataset
                        config["dataset"]["base_model"] = model
                        config["dataset"]["use_gpnn"] = False
                        config["dataset"]["use_augmented_gnn"] = True
                        config["dataset"]["feature_type"] = feature_type
                        config["dataset"]["lo_feature_idx"] = lo_feature_index
                        config["dataset"]["k"] = []
                        config["dataset"]["r"] = []
                        config["dataset"]["s"] = []
                        config["dataset"]["is_vertex_sp_feature"] = True
                        config["dataset"]["normalize_vertex_features"] = norm
                        config["hyperparameters"] = {}
                        config["hyperparameters"]["num_clusters"] = [n_cluster]
                        config["hyperparameters"]["pca_dims"] = pca_dims
                        config["hyperparameters"]["min_cluster_size"] = min_cluster_size
                        config["hyperparameters"]["num_layers"] = num_layers
                        config["hyperparameters"]["num_hidden_channels"] = num_hidden_channels
                        config["hyperparameters"]["num_batch_sizes"] = num_batch_sizes
                        config["hyperparameters"]["num_epochs"] = num_epochs
                        config["hyperparameters"]["lrs"] = lrs
                        config["hyperparameters"]["num_gpnn_layer"] = num_gpnn_layers
                        config["hyperparameters"]["gpnn_channels"] = num_gpnn_channels

                        util.write_metadata_file(path = path, filename = filename, data = config)

# split up by number of layers
def gen_partition_enhanced_config():
    normalize_features = [False, True]
    batch_norms = [False, True]
    
    for model in models:
        for norm in normalize_features:
            for n_cluster in num_clusters:
                for bn in batch_norms:
                    for k in k_vals:
                        for layer in num_layers:
                            if gen_enhanced_gnn:

                                if bn:
                                    if norm:
                                        title = f"{dataset}_{k}-disk_norm_partition-enhanced_{n_cluster}c_enhanced_{layer}l_{model}_bn"
                                    else:
                                        title = f"{dataset}_{k}-disk_partition-enhanced_{n_cluster}c_enhanced_{layer}l_{model}_bn"
                                else:
                                    if norm:
                                        title = f"{dataset}_{k}-disk_norm_partition-enhanced_{n_cluster}c_enhanced_{layer}l_{model}"
                                    else:
                                        title = f"{dataset}_{k}-disk_partition-enhanced_{n_cluster}c_enhanced_{layer}l_{model}"

                                filename = f"{title}.json"

                                path = osp.join(output_root_folder, "partition-enhanced_gnn", f"{k}-disk", f"{n_cluster}-cluster", model)

                                if norm:
                                    if is_linux:
                                        prev_result_path = f"{prev_res_folder}/{f"{dataset}_{k}-disk_norm_clustering_{n_cluster}c_enhanced_{model}"}.json"
                                    else:
                                        prev_result_path = f"{prev_res_folder}\\{f"{dataset}_{k}-disk_norm_clustering_{n_cluster}c_enhanced_{model}"}.json"
                                else:
                                    if is_linux:
                                        prev_result_path = f"{prev_res_folder}/{f"{dataset}_{k}-disk_clustering_{n_cluster}c_enhanced_{model}"}.json"
                                    else:
                                        prev_result_path = f"{prev_res_folder}\\{f"{dataset}_{k}-disk_clustering_{n_cluster}c_enhanced_{model}"}.json"

                                # gen config
                                config = {}
                                config["type"] = "experiment"
                                config["mode"] = "enhanced"
                                config["title"] = title
                                config["prev_result_path"] = prev_result_path
                                config["general"] = {}
                                config["general"]["seed"] = constants.SEED
                                config["general"]["num_workers"] = constants.num_workers
                                config["general"]["num_reruns"] = num_reruns
                                config["general"]["max_patience"] = max_patience
                                config["general"]["use_batch_norm"] = bn
                                config["general"]["num_k_fold"] = num_k_fold
                                config["general"]["k_fold_test_ratio"] = k_fold_test_ratio
                                config["general"]["mbk_batch_size"] = mbk_batch_size
                                config["general"]["mbk_num_init"] = mbk_num_init
                                config["general"]["mbk_max_no_improvement"] = mbk_max_no_improvement
                                config["general"]["mbk_max_iter"] = mbk_max_iter

                                config["dataset"] = {}
                                config["dataset"]["dataset_str"] = dataset
                                config["dataset"]["base_model"] = model
                                config["dataset"]["use_gpnn"] = False
                                config["dataset"]["use_augmented_gnn"] = False
                                config["dataset"]["feature_type"] = feature_type
                                config["dataset"]["lo_feature_idx"] = lo_feature_index
                                config["dataset"]["k"] = [k]
                                config["dataset"]["r"] = []
                                config["dataset"]["s"] = []
                                config["dataset"]["is_vertex_sp_feature"] = False
                                config["dataset"]["normalize_vertex_features"] = norm
                                config["hyperparameters"] = {}
                                config["hyperparameters"]["num_clusters"] = [n_cluster]
                                config["hyperparameters"]["pca_dims"] = pca_dims
                                config["hyperparameters"]["min_cluster_size"] = min_cluster_size
                                config["hyperparameters"]["num_layers"] = [layer]
                                config["hyperparameters"]["num_hidden_channels"] = num_hidden_channels
                                config["hyperparameters"]["num_batch_sizes"] = num_batch_sizes
                                config["hyperparameters"]["num_epochs"] = num_epochs
                                config["hyperparameters"]["lrs"] = lrs
                                config["hyperparameters"]["num_gpnn_layer"] = num_gpnn_layers
                                config["hyperparameters"]["gpnn_channels"] = num_gpnn_channels

                                util.write_metadata_file(path = path, filename = filename, data = config)

                            if gen_augmented_gnn:
                                if bn:
                                    if norm:
                                        title = f"{dataset}_{k}-disk_norm_partition-enhanced_{n_cluster}c_augmented_{layer}l_{model}_bn"
                                    else:
                                        title = f"{dataset}_{k}-disk_partition-enhanced_{n_cluster}c_augmented_{layer}l_{model}_bn"
                                else:
                                    if norm:
                                        title = f"{dataset}_{k}-disk_norm_partition-enhanced_{n_cluster}c_augmented_{layer}l_{model}"
                                    else:
                                        title = f"{dataset}_{k}-disk_partition-enhanced_{n_cluster}c_augmented_{layer}l_{model}"

                                filename = f"{title}.json"

                                path = osp.join(output_root_folder, "partition-augmented_gnn", f"{k}-disk", f"{n_cluster}-cluster", model)

                                if norm:
                                    if is_linux:
                                        prev_result_path = f"{prev_res_folder}/{f"{dataset}_{k}-disk_norm_clustering_{n_cluster}c_augmented_{model}"}.json"
                                    else:
                                        prev_result_path = f"{prev_res_folder}\\{f"{dataset}_{k}-disk_norm_clustering_{n_cluster}c_augmented_{model}"}.json"
                                else:
                                    if is_linux:
                                        prev_result_path = f"{prev_res_folder}/{f"{dataset}_{k}-disk_clustering_{n_cluster}c_augmented_{model}"}.json"
                                    else:
                                        prev_result_path = f"{prev_res_folder}\\{f"{dataset}_{k}-disk_clustering_{n_cluster}c_augmented_{model}"}.json"

                                # gen config
                                config = {}
                                config["type"] = "experiment"
                                config["mode"] = "enhanced"
                                config["title"] = title
                                config["prev_result_path"] = prev_result_path
                                config["general"] = {}
                                config["general"]["seed"] = constants.SEED
                                config["general"]["num_workers"] = constants.num_workers
                                config["general"]["num_reruns"] = num_reruns
                                config["general"]["max_patience"] = max_patience
                                config["general"]["use_batch_norm"] = bn
                                config["general"]["num_k_fold"] = num_k_fold
                                config["general"]["k_fold_test_ratio"] = k_fold_test_ratio
                                config["general"]["mbk_batch_size"] = mbk_batch_size
                                config["general"]["mbk_num_init"] = mbk_num_init
                                config["general"]["mbk_max_no_improvement"] = mbk_max_no_improvement
                                config["general"]["mbk_max_iter"] = mbk_max_iter

                                config["dataset"] = {}
                                config["dataset"]["dataset_str"] = dataset
                                config["dataset"]["base_model"] = model
                                config["dataset"]["use_gpnn"] = False
                                config["dataset"]["use_augmented_gnn"] = True
                                config["dataset"]["feature_type"] = feature_type
                                config["dataset"]["lo_feature_idx"] = lo_feature_index
                                config["dataset"]["k"] = [k]
                                config["dataset"]["r"] = []
                                config["dataset"]["s"] = []
                                config["dataset"]["is_vertex_sp_feature"] = False
                                config["dataset"]["normalize_vertex_features"] = norm
                                config["hyperparameters"] = {}
                                config["hyperparameters"]["num_clusters"] = [n_cluster]
                                config["hyperparameters"]["pca_dims"] = pca_dims
                                config["hyperparameters"]["min_cluster_size"] = min_cluster_size
                                config["hyperparameters"]["num_layers"] = [layer]
                                config["hyperparameters"]["num_hidden_channels"] = num_hidden_channels
                                config["hyperparameters"]["num_batch_sizes"] = num_batch_sizes
                                config["hyperparameters"]["num_epochs"] = num_epochs
                                config["hyperparameters"]["lrs"] = lrs
                                config["hyperparameters"]["num_gpnn_layer"] = num_gpnn_layers
                                config["hyperparameters"]["gpnn_channels"] = num_gpnn_channels

                                util.write_metadata_file(path = path, filename = filename, data = config)

                            for idx, _ in enumerate(r_vals):
                                if gen_enhanced_gnn:

                                    if bn:
                                        if norm:
                                            title = f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_norm_partition-enhanced_{n_cluster}c_enhanced_{layer}l_{model}_bn"
                                        else:
                                            title = f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_partition-enhanced_{n_cluster}c_enhanced_{layer}l_{model}_bn"
                                    else:
                                        if norm:
                                            title = f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_norm_partition-enhanced_{n_cluster}c_enhanced_{layer}l_{model}"
                                        else:
                                            title = f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_partition-enhanced_{n_cluster}c_enhanced_{layer}l_{model}"

                                    filename = f"{title}.json"

                                    path = osp.join(output_root_folder, "partition-enhanced_gnn", f"{r_vals[idx]}-{s_vals[idx]}-ring", f"{n_cluster}-cluster", model)

                                    if norm:
                                        if is_linux:
                                            prev_result_path = f"{prev_res_folder}/{f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_norm_clustering_{n_cluster}c_enhanced_{model}"}.json"
                                        else:
                                            prev_result_path = f"{prev_res_folder}\\{f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_norm_clustering_{n_cluster}c_enhanced_{model}"}.json"
                                    else:
                                        if is_linux:
                                            prev_result_path = f"{prev_res_folder}/{f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_clustering_{n_cluster}c_enhanced_{model}"}.json"
                                        else:
                                            prev_result_path = f"{prev_res_folder}\\{f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_clustering_{n_cluster}c_enhanced_{model}"}.json"

                                    # gen config
                                    config = {}
                                    config["type"] = "experiment"
                                    config["mode"] = "enhanced"
                                    config["title"] = title
                                    config["prev_result_path"] = prev_result_path
                                    config["general"] = {}
                                    config["general"]["seed"] = constants.SEED
                                    config["general"]["num_workers"] = constants.num_workers
                                    config["general"]["num_reruns"] = num_reruns
                                    config["general"]["max_patience"] = max_patience
                                    config["general"]["use_batch_norm"] = bn
                                    config["general"]["num_k_fold"] = num_k_fold
                                    config["general"]["k_fold_test_ratio"] = k_fold_test_ratio
                                    config["general"]["mbk_batch_size"] = mbk_batch_size
                                    config["general"]["mbk_num_init"] = mbk_num_init
                                    config["general"]["mbk_max_no_improvement"] = mbk_max_no_improvement
                                    config["general"]["mbk_max_iter"] = mbk_max_iter

                                    config["dataset"] = {}
                                    config["dataset"]["dataset_str"] = dataset
                                    config["dataset"]["base_model"] = model
                                    config["dataset"]["use_gpnn"] = False
                                    config["dataset"]["use_augmented_gnn"] = False
                                    config["dataset"]["feature_type"] = feature_type
                                    config["dataset"]["lo_feature_idx"] = lo_feature_index
                                    config["dataset"]["k"] = []
                                    config["dataset"]["r"] = [r_vals[idx]]
                                    config["dataset"]["s"] = [s_vals[idx]]
                                    config["dataset"]["is_vertex_sp_feature"] = False
                                    config["dataset"]["normalize_vertex_features"] = norm
                                    config["hyperparameters"] = {}
                                    config["hyperparameters"]["num_clusters"] = [n_cluster]
                                    config["hyperparameters"]["pca_dims"] = pca_dims
                                    config["hyperparameters"]["min_cluster_size"] = min_cluster_size
                                    config["hyperparameters"]["num_layers"] = [layer]
                                    config["hyperparameters"]["num_hidden_channels"] = num_hidden_channels
                                    config["hyperparameters"]["num_batch_sizes"] = num_batch_sizes
                                    config["hyperparameters"]["num_epochs"] = num_epochs
                                    config["hyperparameters"]["lrs"] = lrs
                                    config["hyperparameters"]["num_gpnn_layer"] = num_gpnn_layers
                                    config["hyperparameters"]["gpnn_channels"] = num_gpnn_channels

                                    util.write_metadata_file(path = path, filename = filename, data = config)

                                if gen_augmented_gnn:
                                    if bn:
                                        if norm:
                                            title = f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_norm_partition-enhanced_{n_cluster}c_augmented_{layer}l_{model}_bn"
                                        else:
                                            title = f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_partition-enhanced_{n_cluster}c_augmented_{layer}l_{model}_bn"
                                    else:
                                        if norm:
                                            title = f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_norm_partition-enhanced_{n_cluster}c_augmented_{layer}l_{model}"
                                        else:
                                            title = f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_partition-enhanced_{n_cluster}c_augmented_{layer}l_{model}"

                                    filename = f"{title}.json"

                                    path = osp.join(output_root_folder, "partition-augmented_gnn", f"{r_vals[idx]}-{s_vals[idx]}-ring", f"{n_cluster}-cluster", model)

                                    if norm:
                                        if is_linux:
                                            prev_result_path = f"{prev_res_folder}/{f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_norm_clustering_{n_cluster}c_augmented_{model}"}.json"
                                        else:
                                            prev_result_path = f"{prev_res_folder}\\{f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_norm_clustering_{n_cluster}c_augmented_{model}"}.json"
                                    else:
                                        if is_linux:
                                            prev_result_path = f"{prev_res_folder}/{f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_clustering_{n_cluster}c_augmented_{model}"}.json"
                                        else:
                                            prev_result_path = f"{prev_res_folder}\\{f"{dataset}_{r_vals[idx]}-{s_vals[idx]}-ring_clustering_{n_cluster}c_augmented_{model}"}.json"

                                    # gen config
                                    config = {}
                                    config["type"] = "experiment"
                                    config["mode"] = "enhanced"
                                    config["title"] = title
                                    config["prev_result_path"] = prev_result_path
                                    config["general"] = {}
                                    config["general"]["seed"] = constants.SEED
                                    config["general"]["num_workers"] = constants.num_workers
                                    config["general"]["num_reruns"] = num_reruns
                                    config["general"]["max_patience"] = max_patience
                                    config["general"]["use_batch_norm"] = bn
                                    config["general"]["num_k_fold"] = num_k_fold
                                    config["general"]["k_fold_test_ratio"] = k_fold_test_ratio
                                    config["general"]["mbk_batch_size"] = mbk_batch_size
                                    config["general"]["mbk_num_init"] = mbk_num_init
                                    config["general"]["mbk_max_no_improvement"] = mbk_max_no_improvement
                                    config["general"]["mbk_max_iter"] = mbk_max_iter

                                    config["dataset"] = {}
                                    config["dataset"]["dataset_str"] = dataset
                                    config["dataset"]["base_model"] = model
                                    config["dataset"]["use_gpnn"] = False
                                    config["dataset"]["use_augmented_gnn"] = True
                                    config["dataset"]["feature_type"] = feature_type
                                    config["dataset"]["lo_feature_idx"] = lo_feature_index
                                    config["dataset"]["k"] = []
                                    config["dataset"]["r"] = [r_vals[idx]]
                                    config["dataset"]["s"] = [s_vals[idx]]
                                    config["dataset"]["is_vertex_sp_feature"] = False
                                    config["dataset"]["normalize_vertex_features"] = norm
                                    config["hyperparameters"] = {}
                                    config["hyperparameters"]["num_clusters"] = [n_cluster]
                                    config["hyperparameters"]["pca_dims"] = pca_dims
                                    config["hyperparameters"]["min_cluster_size"] = min_cluster_size
                                    config["hyperparameters"]["num_layers"] = [layer]
                                    config["hyperparameters"]["num_hidden_channels"] = num_hidden_channels
                                    config["hyperparameters"]["num_batch_sizes"] = num_batch_sizes
                                    config["hyperparameters"]["num_epochs"] = num_epochs
                                    config["hyperparameters"]["lrs"] = lrs
                                    config["hyperparameters"]["num_gpnn_layer"] = num_gpnn_layers
                                    config["hyperparameters"]["gpnn_channels"] = num_gpnn_channels

                                    util.write_metadata_file(path = path, filename = filename, data = config)

                            if vertex_sp:
                                if gen_enhanced_gnn:

                                    if bn:
                                        if norm:
                                            title = f"{dataset}_vsp_norm_partition-enhanced_{n_cluster}c_enhanced_{layer}l_{model}_bn"
                                        else:
                                            title = f"{dataset}_vsp_partition-enhanced_{n_cluster}c_enhanced_{layer}l_{model}_bn"
                                    else:
                                        if norm:
                                            title = f"{dataset}_vsp_norm_partition-enhanced_{n_cluster}c_enhanced_{layer}l_{model}"
                                        else:
                                            title = f"{dataset}_vsp_partition-enhanced_{n_cluster}c_enhanced_{layer}l_{model}"

                                    filename = f"{title}.json"

                                    path = osp.join(output_root_folder, "partition-enhanced_gnn", f"vsp", f"{n_cluster}-cluster", model)

                                    if norm:
                                        if is_linux:
                                            prev_result_path = f"{prev_res_folder}/{f"{dataset}_vsp_norm_clustering_{n_cluster}c_enhanced_{model}"}.json"
                                        else:
                                            prev_result_path = f"{prev_res_folder}\\{f"{dataset}_vsp_norm_clustering_{n_cluster}c_enhanced_{model}"}.json"
                                    else:
                                        if is_linux:
                                            prev_result_path = f"{prev_res_folder}/{f"{dataset}_vsp_clustering_{n_cluster}c_enhanced_{model}"}.json"
                                        else:
                                            prev_result_path = f"{prev_res_folder}\\{f"{dataset}_vsp_clustering_{n_cluster}c_enhanced_{model}"}.json"

                                    # gen config
                                    config = {}
                                    config["type"] = "experiment"
                                    config["mode"] = "enhanced"
                                    config["title"] = title
                                    config["prev_result_path"] = prev_result_path
                                    config["general"] = {}
                                    config["general"]["seed"] = constants.SEED
                                    config["general"]["num_workers"] = constants.num_workers
                                    config["general"]["num_reruns"] = num_reruns
                                    config["general"]["max_patience"] = max_patience
                                    config["general"]["use_batch_norm"] = bn
                                    config["general"]["num_k_fold"] = num_k_fold
                                    config["general"]["k_fold_test_ratio"] = k_fold_test_ratio
                                    config["general"]["mbk_batch_size"] = mbk_batch_size
                                    config["general"]["mbk_num_init"] = mbk_num_init
                                    config["general"]["mbk_max_no_improvement"] = mbk_max_no_improvement
                                    config["general"]["mbk_max_iter"] = mbk_max_iter

                                    config["dataset"] = {}
                                    config["dataset"]["dataset_str"] = dataset
                                    config["dataset"]["base_model"] = model
                                    config["dataset"]["use_gpnn"] = False
                                    config["dataset"]["use_augmented_gnn"] = False
                                    config["dataset"]["feature_type"] = feature_type
                                    config["dataset"]["lo_feature_idx"] = lo_feature_index
                                    config["dataset"]["k"] = []
                                    config["dataset"]["r"] = []
                                    config["dataset"]["s"] = []
                                    config["dataset"]["is_vertex_sp_feature"] = True
                                    config["dataset"]["normalize_vertex_features"] = norm
                                    config["hyperparameters"] = {}
                                    config["hyperparameters"]["num_clusters"] = [n_cluster]
                                    config["hyperparameters"]["pca_dims"] = pca_dims
                                    config["hyperparameters"]["min_cluster_size"] = min_cluster_size
                                    config["hyperparameters"]["num_layers"] = [layer]
                                    config["hyperparameters"]["num_hidden_channels"] = num_hidden_channels
                                    config["hyperparameters"]["num_batch_sizes"] = num_batch_sizes
                                    config["hyperparameters"]["num_epochs"] = num_epochs
                                    config["hyperparameters"]["lrs"] = lrs
                                    config["hyperparameters"]["num_gpnn_layer"] = num_gpnn_layers
                                    config["hyperparameters"]["gpnn_channels"] = num_gpnn_channels

                                    util.write_metadata_file(path = path, filename = filename, data = config)

                                if gen_augmented_gnn:
                                    if bn:
                                        if norm:
                                            title = f"{dataset}_vsp_norm_partition-enhanced_{n_cluster}c_augmented_{layer}l_{model}_bn"
                                        else:
                                            title = f"{dataset}_vsp_partition-enhanced_{n_cluster}c_augmented_{layer}l_{model}_bn"
                                    else:
                                        if norm:
                                            title = f"{dataset}_vsp_norm_partition-enhanced_{n_cluster}c_augmented_{layer}l_{model}"
                                        else:
                                            title = f"{dataset}_vsp_partition-enhanced_{n_cluster}c_augmented_{layer}l_{model}"

                                    filename = f"{title}.json"

                                    path = osp.join(output_root_folder, "partition-augmented_gnn", f"vsp", f"{n_cluster}-cluster", model)

                                    if norm:
                                        if is_linux:
                                            prev_result_path = f"{prev_res_folder}/{f"{dataset}_vsp_norm_clustering_{n_cluster}c_augmented_{model}"}.json"
                                        else:
                                            prev_result_path = f"{prev_res_folder}\\{f"{dataset}_vsp_norm_clustering_{n_cluster}c_augmented_{model}"}.json"
                                    else:
                                        if is_linux:
                                            prev_result_path = f"{prev_res_folder}/{f"{dataset}_vsp_clustering_{n_cluster}c_augmented_{model}"}.json"
                                        else:
                                            prev_result_path = f"{prev_res_folder}\\{f"{dataset}_vsp_clustering_{n_cluster}c_augmented_{model}"}.json"

                                    # gen config
                                    config = {}
                                    config["type"] = "experiment"
                                    config["mode"] = "enhanced"
                                    config["title"] = title
                                    config["prev_result_path"] = prev_result_path
                                    config["general"] = {}
                                    config["general"]["seed"] = constants.SEED
                                    config["general"]["num_workers"] = constants.num_workers
                                    config["general"]["num_reruns"] = num_reruns
                                    config["general"]["max_patience"] = max_patience
                                    config["general"]["use_batch_norm"] = bn
                                    config["general"]["num_k_fold"] = num_k_fold
                                    config["general"]["k_fold_test_ratio"] = k_fold_test_ratio
                                    config["general"]["mbk_batch_size"] = mbk_batch_size
                                    config["general"]["mbk_num_init"] = mbk_num_init
                                    config["general"]["mbk_max_no_improvement"] = mbk_max_no_improvement
                                    config["general"]["mbk_max_iter"] = mbk_max_iter

                                    config["dataset"] = {}
                                    config["dataset"]["dataset_str"] = dataset
                                    config["dataset"]["base_model"] = model
                                    config["dataset"]["use_gpnn"] = False
                                    config["dataset"]["use_augmented_gnn"] = True
                                    config["dataset"]["feature_type"] = feature_type
                                    config["dataset"]["lo_feature_idx"] = lo_feature_index
                                    config["dataset"]["k"] = []
                                    config["dataset"]["r"] = []
                                    config["dataset"]["s"] = []
                                    config["dataset"]["is_vertex_sp_feature"] = True
                                    config["dataset"]["normalize_vertex_features"] = norm
                                    config["hyperparameters"] = {}
                                    config["hyperparameters"]["num_clusters"] = [n_cluster]
                                    config["hyperparameters"]["pca_dims"] = pca_dims
                                    config["hyperparameters"]["min_cluster_size"] = min_cluster_size
                                    config["hyperparameters"]["num_layers"] = [layer]
                                    config["hyperparameters"]["num_hidden_channels"] = num_hidden_channels
                                    config["hyperparameters"]["num_batch_sizes"] = num_batch_sizes
                                    config["hyperparameters"]["num_epochs"] = num_epochs
                                    config["hyperparameters"]["lrs"] = lrs
                                    config["hyperparameters"]["num_gpnn_layer"] = num_gpnn_layers
                                    config["hyperparameters"]["gpnn_channels"] = num_gpnn_channels

                                    util.write_metadata_file(path = path, filename = filename, data = config)


if __name__ == '__main__':
    
    # 0 -> only classical, 1 -> only clustering, 2 -> only enhanced/augmented, 3 -> all stages
    if stage == 0:
        gen_classical_config()
    elif stage == 1:
        gen_clustering_config()
    elif stage == 2:
        gen_partition_enhanced_config()
    elif stage == 3:
        gen_classical_config()
        gen_clustering_config()
        gen_partition_enhanced_config()
    else:
        raise ValueError