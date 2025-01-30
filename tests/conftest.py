# Copyright 2024 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from sklearn.datasets import make_regression, make_classification
import pandas as pd
import os
import json
import subprocess
from typing import Union, Literal, Any
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
        raise ValueError(
            f"problem_type is not regression or classification. value given: {problem_type}"
        )


def dataset_std_def_reg() -> dict[str, Any]:
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


def dataset_std_def_clf_bin() -> dict[str, Any]:
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


def dataset_std_def_clf_mult() -> dict[str, Any]:
    def_dict = dataset_std_def_clf_bin()
    def_dict["n_classes"] = 3
    def_dict["weights"] = [0.5, 0.3]
    return def_dict


def dataset_create_files(
    problem_type: Literal["regression", "classification"],
    multi: bool = False,
    r2g: bool = False,
) -> tuple[str, Union[str, None]]:
    if problem_type == "regression":
        x, y, _ = make_regression(**dataset_std_def_reg())
    elif problem_type in "classification":
        if multi:
            x, y = make_classification(**dataset_std_def_clf_mult())
        else:
            x, y = make_classification(**dataset_std_def_clf_bin())
    else:
        raise ValueError(
            f"problem_type is not regression or classification. value given: {problem_type}"
        )

    if not r2g:
        x_fname = f"data/generated_training_{problem_type}"
        x_fname += (
            "_multi.csv" if (multi and (problem_type == "classification")) else ".csv"
        )

        y_fname = f"data/generated_target_{problem_type}"
        y_fname += (
            "_multi.csv" if (multi and (problem_type == "classification")) else ".csv"
        )

        pd.DataFrame(x).to_csv(x_fname)
        pd.DataFrame(y, columns=["target"]).to_csv(y_fname)
    else:
        x_fname = f"data/generated_r2g_{problem_type}"
        x_fname += (
            "_multi.csv" if (multi and (problem_type == "classification")) else ".csv"
        )

        df = pd.DataFrame(x)
        df["set"] = "train"
        df["set"].iloc[-20:] = "test"
        df["label"] = y

        df.to_csv(x_fname)
        y_fname = None

    return x_fname, y_fname


##################### CONFIG CREATION #####################


def config_autokeras() -> dict[str, Any]:
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


def config_autolgbm() -> dict[str, Any]:
    outdict = {
        "autolgbm_config": {"verbose": True, "n_trials": 15, "timeout": 60},
    }
    return outdict


def config_autoxgboost() -> dict[str, Any]:
    outdict = {
        "autoxgboost_config": {"verbose": True, "n_trials": 10, "timeout": 1000},
    }
    return outdict


def config_all_auto_methods() -> dict[str, Any]:
    outdict = {
        **config_autokeras(),
        **config_autolgbm(),
        **config_autoxgboost(),
    }
    return outdict


def config_plot_classification() -> list[str]:
    outlist = ["conf_matrix", "roc_curve"]
    return outlist


def config_plot_regression() -> list[str]:
    outlist = [
        "hist_overlapped",
        "joint",
        "joint_dens",
        "corr",
    ]
    return outlist


def config_plot_both() -> list[str]:
    outlist = [
        "barplot_scorer",
        "boxplot_scorer",
        "shap_plots",
        "permut_imp_test",
    ]
    return outlist


def config_all_plotting(
    problem_type: Literal["regression", "classification"]
) -> dict[str, Any]:
    outlist = config_plot_both()
    if problem_type == "regression":
        outlist += config_plot_regression()
    elif problem_type == "classification":
        outlist += config_plot_classification()
    else:
        raise ValueError(
            f"problem_type must be regression or classification, recieved: {problem_type}"
        )

    outdict = {
        "plotting": {
            "plot_method": outlist,
            "top_feats_permImp": 20,
            "top_feats_shap": 20,
            "explanations_data": "all",
        }
    }
    return outdict


def config_feature_selection() -> dict[str, Any]:
    outdict = {"feature_selection": {"k": "auto", "auto": {"max_features": None}}}
    return outdict


def config_model_list(
    problem_type: Literal["regression", "classification"]
) -> dict[str, Any]:
    if problem_type == "regression":
        ml = [
            "RandomForestRegressor",
            "SVR",
            "KNeighborsRegressor",
            "AdaBoostRegressor",
            "XGBRegressor",
        ]
    elif problem_type == "classification":
        ml = [
            "RandomForestClassifier",
            "SVC",
            "KNeighborsClassifier",
            "AdaBoostClassifier",
            "XGBClassifier",
        ]
    else:
        raise ValueError(f"Unexpected problem type: {problem_type}")

    outdict = {
        "model_list": ml
        + [
            "AutoKeras",
            "AutoLGBM",
            "AutoXGBoost",
            "FixedKeras",
        ]
    }
    return outdict


def config_scorers(
    problem_type: Literal["regression", "classification"]
) -> dict[str, Any]:
    if problem_type == "regression":
        outdict = {
            "fit_scorer": "mean_absolute_percentage_error",
            "scorer_list": [
                "explained_variance_score",
                "mean_squared_error",
                "rmse",
                "mean_absolute_error",
                "median_absolute_error",
                "mean_absolute_percentage_error",
                "r2_score",
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
    else:
        raise ValueError(
            f"problem_type must be regression or classification, recieved: {problem_type}"
        )
    return outdict


def config_data_paths(
    file_path: str,
    meta_path: str,
    problem_type: str,
    multi: bool = False,
    r2g: bool = False,
) -> dict[str, Union[str, None]]:
    outdict = {
        "data": {
            "name": ("ready_" if r2g else "")
            + "generated_test_"
            + problem_type
            + ("_multi" if multi else "")
            + "_run1_",
            "file_path": "/" + file_path,
            "metadata_file": None if r2g else ("/" + meta_path),
            "file_path_holdout_data": "/" + file_path,
            "metadata_file_holdout_data": None if r2g else ("/" + meta_path),
            "save_path": "/experiments/",
            "target": "label" if r2g else "target",
            "data_type": "R2G" if r2g else "other",
        }
    }
    return outdict


def config_prediction(file_path: str) -> dict[str, dict[str, str]]:
    outdict = {"prediction": {"file_path": "/" + file_path}}
    return outdict


def config_define_problem(
    problem_type: Literal["regression", "classification"]
) -> dict[str, Any]:
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


def config_define_ml(
    problem_type: Literal["regression", "classification"]
) -> dict[str, Any]:
    outdict = {
        "ml": {
            **config_define_problem(problem_type),
            **config_scorers(problem_type),
            **config_model_list(problem_type),
            **config_all_auto_methods(),
            **config_feature_selection(),
        }
    }

    return outdict


def config_microbiome() -> dict[str, None]:
    outdict = {
        "collapse_tax": None,
        "min_reads": None,
        "norm_reads": None,
        "filter_abundance": None,
        "filter_prevalence": None,
        "filter_microbiome_samples": None,
        "remove_classes": None,
        "merge_classes": None,
    }
    return outdict


def config_gene_expression() -> dict[str, None]:
    outdict = {
        "expression_type": None,
        "filter_sample": None,
        "filter_genes": None,
        "output_file_ge": None,
        "output_metadata": None,
    }
    return outdict


def config_metabolmic() -> dict[str, None]:
    outdict = {
        "filter_metabolomic_sample": None,
        "filter_measurements": None,
        "output_file_met": None,
        "output_metadata": None,
    }
    return outdict


def config_tabular() -> dict[str, None]:
    outdict = {
        "filter_tabular_sample": None,
        "filter_tabular_measurements": None,
        "output_file_tab": None,
        "output_metadata": None,
    }
    return outdict


def config_create(
    problem_type: Literal["regression", "classification"],
    file_path: str,
    meta_path: Union[str, None],
    multi: bool = False,
    run: int = 1,
    r2g: bool = False,
) -> str:
    outdict = {
        **config_data_paths(file_path, meta_path, problem_type, multi, r2g=r2g),
        **config_all_plotting(problem_type),
        **config_define_ml(problem_type),
        **config_prediction(file_path),
        # **config_microbiome(),                      # settings for the corresponding data type
        # **config_gene_expression(),                 # settings for the corresponding data type
        # **config_metabolmic(),                      # settings for the corresponding data type
        # **config_tabular(),                         # settings for the corresponding data type
    }

    outdict["data"]["name"] += str(run).zfill(len(str(RUNS)))
    fname = f"configs/generated_test_{problem_type}"
    fname += (
        "_multi.json" if (multi and (problem_type == "classification")) else ".json"
    )

    with open(fname, "w") as outfile:
        json.dump(outdict, outfile, indent=4)

    return fname


##################### TEST PROBLEM CREATION #####################
STARTS = 0
RUNS = 1


# Fixture for creating datasets & jsons to test on
@pytest.fixture(scope="session")
def problem_create(request):
    problem_type = request.param[0]
    problem_type = "classification" if problem_type == "binary" else problem_type
    problem_type = "regression" if problem_type == "reg" else problem_type
    r2g = request.param[1]
    if problem_type == "multi":
        x_fname, y_fname = dataset_create_files("classification", multi=True, r2g=r2g)
        fname = config_create("classification", x_fname, y_fname, multi=True, r2g=r2g)
    else:
        x_fname, y_fname = dataset_create_files(problem_type, r2g=r2g)
        fname = config_create(problem_type, x_fname, y_fname, r2g=r2g)

    yield fname

    if x_fname:
        os.remove(x_fname)
    if y_fname:
        os.remove(y_fname)
    os.remove(fname)
    # shutil.rmtree(f'experiments/results/{fname[:-5]}_run1_{str(run).zfill(len(str(RUNS)))}')


######### Fixture to build container #######
@pytest.fixture(scope="session")
def container() -> bool:
    sp = subprocess.call(["./build.sh"])
    return sp == 0
