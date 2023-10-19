import pytest
from sklearn.datasets import make_regression, make_classification
import pandas as pd
import os
import json

import shutil

##################### DATA SET CREATION #####################


def dataset_create_data(problem_type, **kwargs):
    if problem_type == "regression":
        x, y, coef = make_regression(**kwargs)
        return x, y, coef

    elif problem_type == "classification":
        x, y = make_classification(**kwargs)
        return x, y

    else:
        raise ValueError(f"problem_type is not regression or classification. value given: {problem_type}")


def dataset_std_def_reg():
    def_dict = {
        "n_samples": 100,
        "n_features": 200,
        "n_informative": 10,
        "n_targets": 1,
        "shuffle": True,
        "coef": True,
        "random_state": 29292,
    }
    return def_dict


def dataset_std_def_clf_bin():
    def_dict = {
        "n_samples": 100,
        "n_features": 200,
        "n_informative": 10,
        "n_classes": 2,
        "n_clusters_per_class": 1,
        "weights": [0.6],
        "flip_y": 0,
        "shuffle": True,
        "random_state": 29292,
    }
    return def_dict


def dataset_std_def_clf_mult():
    def_dict = dataset_std_def_clf_bin()
    def_dict["n_classes"] = 3
    def_dict["weights"] = [0.5, 0.3]
    return def_dict


def dataset_create_files(problem_type, multi=False):
    if problem_type == "regression":
        x, y, _ = make_regression(**dataset_std_def_reg())
    elif problem_type == "classification":
        if multi:
            x, y = make_classification(**dataset_std_def_clf_mult())
        else:
            x, y = make_classification(**dataset_std_def_clf_bin())
    else:
        raise ValueError(f"problem_type is not regression or classification. value given: {problem_type}")

    x_fname = f"data/generated_training_{problem_type}"
    y_fname = f"data/generated_target_{problem_type}"

    x_fname += "_multi.csv" if (multi and (problem_type == "classification")) else ".csv"
    y_fname += "_multi.csv" if (multi and (problem_type == "classification")) else ".csv"

    pd.DataFrame(x).to_csv(x_fname)
    pd.DataFrame(y, columns=["target"]).to_csv(y_fname)

    return x_fname, y_fname


##################### CONFIG CREATION #####################


def config_autosklearn():
    outdict = {
        "autosklearn_config": {
            "verbose": True,
            "estimators": [
                "decision_tree",
                "extra_trees",
                "k_nearest_neighbors",
                "random_forest",
            ],
            "time_left_for_this_task": 120,
            "per_run_time_limit": 60,
            "memory_limit": None,
            "n_jobs": 1,
            "ensemble_size": 1,
        }
    }
    return outdict


def config_autokeras():
    outdict = {
        "autokeras_config": {
            "n_epochs": 100,
            "batch_size": 32,
            "verbose": True,
            "n_blocks": 3,
            "dropout": 0.3,
            "use_batchnorm": True,
            "n_trials": 4,
            "tuner": "bayesian",
        }
    }
    return outdict


def config_autolgbm():
    outdict = {
        "autolgbm_config": {"verbose": True, "n_trials": 15, "timeout": 60},
    }
    return outdict


def config_autoxgboost():
    outdict = {
        "autoxgboost_config": {"verbose": True, "n_trials": 10, "timeout": 1000},
    }
    return outdict


# def config_autogluon():
#     outdict = {
#         "autogluon_config": {
#             "verbose": True,
#             "auto_stack": False,
#             "time_limits": 120
#         },
#     }
#     return outdict


def config_all_auto_methods():
    outdict = {
        **config_autosklearn(),
        **config_autokeras(),
        **config_autolgbm(),
        **config_autoxgboost(),
    }
    return outdict


def config_plot_classification():
    outlist = ["conf_matrix", "roc_curve"]
    return outlist


def config_plot_regression():
    outlist = [
        "hist_overlapped",
        "joint",
        "joint_dens",
        "corr",
    ]
    return outlist


def config_plot_both():
    outlist = [
        "barplot_scorer",
        "boxplot_scorer",
        "shap_plots",
        "permut_imp_test",
    ]
    return outlist


def config_all_plotting(problem_type):
    outlist = config_plot_both()
    if problem_type == "regression":
        outlist += config_plot_regression()
    elif problem_type == "classification":
        outlist += config_plot_classification()

    outdict = {
        "plotting": {
            "plot_method": outlist,
            "top_feats_permImp": 20,
            "top_feats_shap": 20,
            "explanations_data": "all",
        }
    }
    return outdict


def config_feature_selection():
    outdict = {"feature_selection": {"k": "auto", "auto": {"max_features": None}}}
    return outdict


def config_model_list(problem_type):
    if problem_type == "regression":
        pass
    elif problem_type == "classification":
        pass
    else:
        raise ValueError(f"Unexpected problem type: {problem_type}")
    outdict = {
        "model_list":
        # ml+
        ["AutoKeras", "AutoLGBM", "AutoSKLearn", "AutoXGBoost", "FixedKeras"]
    }
    return outdict


def config_scorers(problem_type):
    if problem_type == "regression":
        outdict = {
            "fit_scorer": "mean_absolute_percentage_error",
            "scorer_list": [
                "explained_variance_score",
                "mean_squared_error",
                # "mean_squared_log_error",
                "rmse",
                "mean_absolute_error",
                "median_absolute_error",
                "mean_absolute_percentage_error",
                "r2_score",
                # "mean_poisson_deviance",
                # "mean_gamma_deviance",
                # "mean_tweedie_deviance",
            ],
        }
    elif problem_type == "classification":
        outdict = {
            "fit_scorer": "f1_score",
            "scorer_list": [
                "accuracy_score",
                "f1_score",
                "hamming_loss",
                "jaccard_score",
                "matthews_corrcoef",
                "precision_score",
                "recall_score",
                "zero_one_loss",
                "roc_auc_score",
            ],
        }
    return outdict


def config_data_paths(file_path, meta_path, problem_type, multi=False):
    outdict = {
        "data": {
            "name": "generated_test_" + problem_type + ("_multi" if multi else "") + "_run1_",
            "file_path": "/" + file_path,
            "metadata_file": "/" + meta_path,
            "file_path_holdout_data": "/" + file_path,
            "metadata_file_holdout_data": "/" + meta_path,
            "save_path": "/experiments/",
            "target": "target",
            "data_type": "other",
        }
    }
    return outdict


def config_prediction(file_path):
    outdict = {"prediction": {"file_path": "/" + file_path}}
    return outdict


def config_define_problem(problem_type):
    outdict = {
        "seed_num": 29292,
        "test_size": 0.2,
        "problem_type": problem_type,
        "hyper_tuning": "random",
        "hyper_budget": 50,
        "stratify_by_groups": "N",
        "groups": "",
        "balancing": "OVER",
    }
    return outdict


def config_define_ml(problem_type):
    outdict = {
        "ml": {
            **config_define_problem(problem_type),  # Define the Problem and general parameters            # [EXP/AUTO]
            **config_scorers(problem_type),  # Define scorers to be used depending on problem type  # [HIDDEN]
            **config_model_list(problem_type),  # Define the models to be trained                      # [ADV]
            **config_all_auto_methods(),  # Define auto methods setting                          # [HIDDEN]
            **config_feature_selection(),  # Define the feature selection settings                # [ADV/AUTO]
        }
    }

    return outdict


def config_microbiome():
    outdict = {
        "collapse_tax": None,  # opt: ["genus" or "species"], list of strs, dft None
        "min_reads": None,  # dft: 1000, int >= 0 (can be None?)
        "norm_reads": None,  # dft: 1000, int >= 0 (can be None?)
        "filter_abundance": None,  # dft: 10, int >=0 (can be None?)
        "filter_prevalence": None,  # float: 0<=x<=1, dft: 0.01 (can be None?)
        "filter_microbiome_samples": None,  # list of dict with col as keys and list of strs to remove as values ------------------- check if understood correctly
        "remove_classes": None,  # list of strs, dft none
        "merge_classes": None,  # dict with keys & list of strs for values, dft none
    }
    return outdict


def config_gene_expression():
    outdict = {
        "expression_type": None,  # one of [FPKM,RPKM,TMM,TPM,Log2FC,COUNTS,OTHER] MANDITORY
        "filter_sample": None,  # dft: 100000, numeric >0 (can be None?)
        "filter_genes": None,  # list of two ints >=0, dft [0,1] ------------------------------ check ints vs strs (can be None?)
        "output_file_ge": None,  # str file name, dft None
        "output_metadata": None,  # str file name, dft None
    }
    return outdict


def config_metabolmic():
    outdict = {
        "filter_metabolomic_sample": None,  # numeric >=0, dft: 100000 (can be None?)
        "filter_measurements": None,  # list of two ints >=0, dft [0,1] ------------------------------ check ints vs strs (can be None?)
        "output_file_met": None,  # str file name, dft None
        "output_metadata": None,  # str file name, dft None
    }
    return outdict


def config_tabular():
    outdict = {
        "filter_tabular_sample": None,  # numeric >=0, dft: 100000 (can be None?)
        "filter_tabular_measurements": None,  # list of two ints >=0, dft [0,1] ------------------------------ check ints vs strs (can be None?)
        "output_file_tab": None,  # str file name, dft None
        "output_metadata": None,  # str file name, dft None
    }
    return outdict


def config_create(problem_type, file_path, meta_path, multi=False, run=1):
    outdict = {
        **config_data_paths(file_path, meta_path, problem_type, multi),  # Define the paths to where the data is stored
        # **config_all_plotting(problem_type),  # Define what plots to do
        **config_define_ml(problem_type),
        **config_prediction(file_path),
        # **config_microbiome(),                      # settings for the corresponding data type
        # **config_gene_expression(),                 # settings for the corresponding data type
        # **config_metabolmic(),                      # settings for the corresponding data type
        # **config_tabular(),                         # settings for the corresponding data type
    }

    outdict["data"]["name"] += str(run).zfill(len(str(RUNS)))
    fname = f"configs/generated_test_{problem_type}"
    fname += "_multi.json" if (multi and (problem_type == "classification")) else ".json"

    with open(fname, "w") as outfile:
        json.dump(outdict, outfile, indent=4)

    return fname


##################### TEST PROBLEM CREATION #####################
STARTS = 0
RUNS = 1


@pytest.fixture(
    params=[pytest.param(("classification", i), marks=pytest.mark.classification) for i in range(STARTS + 1, RUNS + 1)]
    + [
        pytest.param(("multi", i), marks=[pytest.mark.classification, pytest.mark.multi])
        for i in range(STARTS + 1, RUNS + 1)
    ]
    + [pytest.param(("regression", i), marks=pytest.mark.regression) for i in range(STARTS + 1, RUNS + 1)],
    scope="session",
)
def problem_create(request):
    problem_type = request.param[0]
    run = request.param[1]
    if problem_type == "multi":
        x_fname, y_fname = dataset_create_files("classification", multi=True)
        fname = config_create("classification", x_fname, y_fname, multi=True, run=run)
    else:
        x_fname, y_fname = dataset_create_files(problem_type)
        fname = config_create(problem_type, x_fname, y_fname, run=run)

    yield fname

    os.remove(x_fname)
    os.remove(y_fname)
    os.remove(fname)
    # shutil.rmtree(f'experiments/results/{fname[:-5]}_run1_{str(run).zfill(len(str(RUNS)))}')
