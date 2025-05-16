# Quick script to generate a latex result table from experiment result files

import util
import os.path as osp
import os
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, getcontext

k_vals = [3, 5] # [3]
r_vals = [3, 3] # [1, 2]
s_vals = [5, 10]
vsp = True
models = ["gin", "gcn"]
n_cluster_enh = [2,3,4]
n_cluster_aug = [2,3,4,10,20]
n_layers = [2, 3, 5]
dataset = "PROTEINS"

factor = 100

include_bn = False
models_in_one_table = len(models) <= 2 and not include_bn

full_run = False

results_path = f"E:\\ResultsBackup\\{dataset}" # "E:\\ResultsBackup\\CSL"

output_path = "E:\\workspace\\PartitionEnhancedGNNsTesting\\experiments\\results\\tables_gen"

if __name__ == '__main__':
    assert len(k_vals) > 0 or len(r_vals) > 0 or vsp

    if not full_run:
        include_bn = False

    getcontext().rounding = ROUND_HALF_UP

    # get all files from directory and subdirectories

    for model in models:
        if models_in_one_table:
            output_filename = f"table_{dataset}.txt"
        else:
            output_filename = f"{model}_table_{dataset}.txt"

        # get classical model res

        if len(n_cluster_enh) > len(n_cluster_aug):
            clusters_bigger = n_cluster_enh
        else:
            clusters_bigger = n_cluster_aug

        num_per_feature_type = 2 * len(clusters_bigger)

        if vsp:
            num_vsp = num_per_feature_type
        
        num_k_disks = len(k_vals) * num_per_feature_type
        num_r_s_rings = len(r_vals) * num_per_feature_type

        num_features = num_vsp + num_k_disks + num_r_s_rings

        if models_in_one_table:
            num_cols = 3 * len(models) 
            if include_bn:
                num_cols*=2
        else:
            if include_bn:
                num_cols = 6
            else:
                num_cols = 3

        table_means = np.full(shape = (num_features, num_cols), dtype = object, fill_value = -1)
        table_stds = np.full(shape = (num_features, num_cols), dtype = object, fill_value = -1)

        if full_run:
            table_means_layer = [[np.full(shape = (num_features, num_cols), dtype = object, fill_value = -1), np.full(shape = (num_features, num_cols), dtype = np.float64, fill_value = -1)] for _ in n_layers]
            table_stds_layer = [np.full(shape = (num_features, num_cols), dtype = object, fill_value = -1) for _ in n_layers]

        

        # for c in n_cluster:
        #     if vsp:
        #         if full_run:
        #             filenames_enh = [f"{dataset}_vsp_partition-enhanced_{c}c_enhanced_{l}l_{model}.json" for l in n_layers]
        #             filenames_norm_enh = [f"{dataset}_vsp_norm_partition-enhanced_{c}c_enhanced_{l}l_{model}.json" for l in n_layers for l in n_layers]
        #             filenames_aug = [f"{dataset}_vsp_partition-enhanced_{c}c_augmented_{l}l_{model}.json" for l in n_layers]
        #             filenames_norm_aug = [f"{dataset}_vsp_norm_partition-enhanced_{c}c_augmented_{l}l_{model}.json" for l in n_layers for l in n_layers]
        #             filenames_enh_bn = [f"{dataset}_vsp_partition-enhanced_{c}c_enhanced_{l}l_{model}_bn.json" for l in n_layers]
        #             filenames_norm_enh_bn = [f"{dataset}_vsp_norm_partition-enhanced_{c}c_enhanced_{l}l_{model}_bn.json" for l in n_layers for l in n_layers]
        #             filenames_aug_bn = [f"{dataset}_vsp_partition-enhanced_{c}c_augmented_{l}l_{model}_bn.json" for l in n_layers]
        #             filenames_norm_aug_bn = [f"{dataset}_vsp_norm_partition-enhanced_{c}c_augmented_{l}l_{model}_bn.json" for l in n_layers for l in n_layers]
        #         else:
        #             filename = f"{dataset}_"
            
        feature_types = []
        if vsp:
            feature_types.append(tuple(["vsp", 0]))
        for idx_ring in range(len(r_vals)):
            feature_types.append(tuple([f"{r_vals[idx_ring]}-{s_vals[idx_ring]}-ring", idx_ring]))
        for idx_disk, k in enumerate(k_vals):
            feature_types.append(tuple([f"{k}-disk", idx_disk]))

        row_labels = ["" for _ in range(num_features)]

        

        for root, dirs, files in os.walk(results_path):
            for file in files:
                for model_inner in models:
                    if (not models_in_one_table) and model_inner.endswith("gcn"):
                        continue

                    classical_filename = f"{dataset}_classical_{model_inner}.json"
                    classical_bn_filename = f"{dataset}_classical_{model_inner}_bn.json"
                    if file.endswith(classical_filename):
                        # classical model
                        classical_res = util.read_metadata_file(path = osp.join(root, file))
                        classical_mean = Decimal(classical_res["classic_gnn_hyperparameter_opt"]["res"]["test_acc"]["mean"])
                        classical_std = Decimal(classical_res["classic_gnn_hyperparameter_opt"]["res"]["test_acc"]["std"])
                        
                        gnn_idx = 0
                        if models_in_one_table:
                            if model_inner.endswith("gcn"):
                                gnn_idx += 3
                                if include_bn:
                                    gnn_idx += 3
                            
                        for idx in range(num_features):
                            table_means[idx, gnn_idx] = classical_mean
                            table_stds[idx, gnn_idx] = classical_std
                    elif file.endswith(classical_bn_filename):

                        if include_bn:
                            # classical model
                            classical_res = util.read_metadata_file(path = osp.join(root, file))
                            classical_mean = Decimal(classical_res["classic_gnn_hyperparameter_opt"]["res"]["test_acc"]["mean"])
                            classical_std = Decimal(classical_res["classic_gnn_hyperparameter_opt"]["res"]["test_acc"]["std"])

                            gnn_idx = 0
                            if models_in_one_table:
                                if model_inner.endswith("gcn"):
                                    gnn_idx += 6

                            for idx in range(num_features):
                                table_means[idx,1] = classical_mean
                                table_stds[idx,1] = classical_std
                            continue
                
                finished = False
                for model_inner in models:
                    if (not models_in_one_table) and model_inner.endswith("gcn"):
                        continue

                    for bn in ["_bn", ""]:
                        if finished:
                            break
                        for enhtype in ["enhanced", "augmented"]:
                            if finished:
                                break
                            for norm in ["norm_", ""]:
                                for c_idx, c  in enumerate(clusters_bigger):
                                    if finished:
                                        break
                                    for feature_type, idx in feature_types:
                                        if finished:
                                            break
                                        for l_idx, l in enumerate(n_layers):
                                            if finished:
                                                break
                                            if full_run:
                                                filename = f"{dataset}_{feature_type}_{norm}partition-enhanced_{c}c_{enhtype}_{l}l_{model_inner}{bn}.json"
                                                if file.endswith(filename):
                                                    finished = True
                                                    res = util.read_metadata_file(path = osp.join(root, file))
                                                    # Identify the col and row of the results

                                                    row_label = ""

                                                    cur_row = 0
                                                    if feature_type.endswith("vsp"):
                                                        # vsp feature
                                                        cur_row += 2 * c_idx
                                                        row_label = f"$\\text{{vsp}}({c}\\text{{c}})$"
                                                        if norm == "norm_":
                                                            cur_row += 1
                                                            row_label = f"$\\text{{vsp}}_{{\\text{{norm}}}}({c}\\text{{c}})$"
                                                    elif feature_type.endswith("-disk"):
                                                        # k-disk
                                                        cur_row += num_vsp
                                                        cur_row += 2 * len(clusters_bigger) * idx
                                                        cur_row += 2 * c_idx
                                                        row_label = f"$\\text{{{k_vals[idx]}-disk}}({c}\\text{{c}})$"
                                                        if norm == "norm_":
                                                            cur_row += 1
                                                            row_label = f"$\\text{{{k_vals[idx]}-disk}}_{{\\text{{norm}}}}({c}\\text{{c}})$"
                                                    elif feature_type.endswith("-ring"):
                                                        # r-s-ring
                                                        cur_row += num_vsp + num_k_disks
                                                        cur_row += 2 * len(clusters_bigger) * idx
                                                        cur_row += 2 * c_idx
                                                        row_label = f"$\\text{{{r_vals[idx]}-{s_vals[idx]}-ring}}({c}\\text{{c}})$"
                                                        if norm == "norm_":
                                                            cur_row += 1
                                                            row_label = f"$\\text{{{r_vals[idx]}-{s_vals[idx]}-ring}}_{{\\text{{norm}}}}({c}\\text{{c}})$"

                                                    cur_col = 1
                                                    if include_bn:
                                                        cur_col += 1
                                                    if enhtype == "augmented":
                                                        cur_col += 1
                                                        if include_bn:
                                                            cur_col += 1
                                                    if bn == "_bn":
                                                        cur_col += 1

                                                    row_labels[cur_row] = row_label

                                                    # write the score into the tables
                                                    table_means_layer[l_idx][0][cur_row,cur_col] = Decimal(res["res"]["val_acc"]["mean"])
                                                    table_means_layer[l_idx][1][cur_row,cur_col] = Decimal(res["res"]["test_acc"]["mean"])
                                                    table_stds_layer[l_idx][cur_row,cur_col] = Decimal(res["res"]["test_acc"]["std"])

                                            else:
                                                filename = f"{dataset}_{feature_type}_{norm}clustering_{c}c_{enhtype}_{model_inner}.json"
                                                if file.endswith(filename):
                                                    finished = True
                                                    res = util.read_metadata_file(path = osp.join(root, file))
                                                    
                                                    # Identify the col and row of the results
                                                    row_label = ""

                                                    cur_row = 0
                                                    if feature_type.endswith("vsp"):
                                                        # vsp feature
                                                        cur_row += 2 * c_idx
                                                        row_label = f"$\\text{{vsp}}({c}\\text{{c}})$"
                                                        if norm == "norm_":
                                                            cur_row += 1
                                                            row_label = f"$\\text{{vsp}}_{{\\text{{norm}}}}({c}\\text{{c}})$"
                                                    elif feature_type.endswith("-disk"):
                                                        # k-disk
                                                        cur_row += num_vsp
                                                        cur_row += 2 * len(clusters_bigger) * idx
                                                        cur_row += 2 * c_idx
                                                        row_label = f"$\\text{{{k_vals[idx]}-disk}}({c}\\text{{c}})$"
                                                        if norm == "norm_":
                                                            cur_row += 1
                                                            row_label = f"$\\text{{{k_vals[idx]}-disk}}_{{\\text{{norm}}}}({c}\\text{{c}})$"
                                                    elif feature_type.endswith("-ring"):
                                                        # r-s-ring
                                                        cur_row += num_vsp + num_k_disks
                                                        cur_row += 2 * len(clusters_bigger) * idx
                                                        cur_row += 2 * c_idx
                                                        row_label = f"$\\text{{{r_vals[idx]}-{s_vals[idx]}-ring}}({c}\\text{{c}})$"
                                                        if norm == "norm_":
                                                            cur_row += 1
                                                            row_label = f"$\\text{{{r_vals[idx]}-{s_vals[idx]}-ring}}_{{\\text{{norm}}}}({c}\\text{{c}})$"

                                                    cur_col = 1
                                                    if model_inner.endswith("gcn"):
                                                        if include_bn:
                                                            cur_col += 6
                                                        else:
                                                            cur_col += 3
                                                    if enhtype == "augmented":
                                                        if include_bn:
                                                            cur_col += 2
                                                        else:
                                                            cur_col += 1
                                                    if include_bn:
                                                        if bn == "_bn":
                                                            cur_col += 1

                                                    row_labels[cur_row] = row_label

                                                    # write the score into the tables
                                                    table_means[cur_row,cur_col] = Decimal(res["clustering_hyperparameter_opt"]["res"]["test_acc"]["mean"])
                                                    table_stds[cur_row,cur_col] = Decimal(res["clustering_hyperparameter_opt"]["res"]["test_acc"]["std"])

        # get the best results based on layers if necessary
        if full_run:
            for model_inner in models:
                if not models_in_one_table and model_inner.endswith("gcn"):
                    continue
                for row in range(num_features):
                    for col in range(num_cols):
                        if col == 0:
                            continue
                        if include_bn:
                            if col == 1:
                                continue
                        if model_inner.endswith("gcn"):
                            if include_bn:
                                if col == 6 or col == 7:
                                    continue
                            else:
                                if col == 3:
                                    continue

                        cur_best_val = Decimal(0.0)
                        cur_best_idx = 0
                        for idx_l in range(len(n_layers)):
                            val = table_means_layer[idx_l][0][row, col]
                            if val > cur_best_val:
                                cur_best_val = val
                                cur_best_idx = idx_l
                    
                        table_means[row, col] = table_means_layer[cur_best_idx][1][row, col]
                        table_stds[row, col] = table_stds_layer[cur_best_idx][row, col]

        # Generate the table
        if models_in_one_table:
            beginning = f"\\begin{{tabular}}{{| l |"
            output = ""
            for model_inner in models:
                if include_bn:
                    output = f"{output} & $\\text{{{model_inner.upper()}}}$ & $\\text{{{model_inner.upper()}}}_{{\\text{{bn}}}}$ & $\\text{{{model_inner.upper()}-enh}}$ & $\\text{{{model_inner.upper()}-enh}}_{{\\text{{bn}}}}$ & $\\text{{{model_inner.upper()}-aug}}$ & $\\text{{{model_inner.upper()}-aug}}_{{\\text{{bn}}}}$"
                    beginning = f"{beginning} c c c c c c"
                else:
                    output = f"{output} & $\\text{{{model_inner.upper()}}}$ & $\\text{{{model_inner.upper()}-enh}}$ & $\\text{{{model_inner.upper()}-aug}}$"
                    beginning = f"{beginning} c c c"
            output = f"{beginning} |}}\n\\hline\nFeature type{output}\\\\\n\\hline"
        else:
            if include_bn:
                output = f"\\begin{{tabular}}{{| l | c c c c c c |}}\n\\hline\nFeature type & $\\text{{{model.upper()}}}$ & $\\text{{{model.upper()}}}_{{\\text{{bn}}}}$ & $\\text{{{model.upper()}-enh}}$ & $\\text{{{model.upper()}-enh}}_{{\\text{{bn}}}}$ & $\\text{{{model.upper()}-aug}}$ & $\\text{{{model.upper()}-aug}}_{{\\text{{bn}}}}$\\\\\n\\hline"
            else:
                output = f"\\begin{{tabular}}{{| l | c c c |}}\n\\hline\nFeature type & $\\text{{{model.upper()}}}$ & $\\text{{{model.upper()}-enh}}$ & $\\text{{{model.upper()}-aug}}$ \\\\\n\\hline"

        count = 0
        for row in range(num_features):
            output = f"{output}\n{row_labels[row]}"
            # get the indices of best performance per row
            cur_best_perf = 0
            best_cols = []
            for col in range(num_cols):
                mean = table_means[row, col]
                if mean > cur_best_perf:
                    cur_best_perf = mean
                    best_cols = [col]
                elif mean == cur_best_perf:
                    best_cols.append(col)

            for col in range(num_cols):
                mean = table_means[row, col]
                if mean == -1:
                    output = f"{output} & "
                else:
                    if col in best_cols:
                        output = f"{output} & $\\mathbf{{{round(mean * factor, 1)}\\pm {round(table_stds[row, col] * factor, 1)}}}$"
                    else:
                        output = f"{output} & ${round(mean * factor, 1)}\\pm {round(table_stds[row, col] * factor, 1)}$"
            output = f"{output} \\\\"
            count += 1
            if count == num_per_feature_type and not (row == num_features - 1):
                output = f"{output}\n\\hline"
            count %= num_per_feature_type
        output = f"{output}\n\\hline\n\\end{{tabular}}"

        util.write_txt(path = output_path, filename = output_filename, text = output)

        if models_in_one_table:
            break