# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pytest
from copy import deepcopy
from ..config_model import ConfigModel as Model

TEST_CONFIG = {
    "data": {
        "name": "50k_barley_rowtype_binary_SHAP",
        "file_path": "data/geno_row_type_BRIDGE_50k_w.hetero.csv",
        "metadata_file": "data/row_type_BRIDGE_pheno_50k_metadata_w.hetero.csv",
        "save_path": "experiments/",
        "target": "row_type_BRIDGE",
        "data_type": "tabular",
    },
    "ml": {
        "seed_num": 42,
        "test_size": 0.2,
        "problem_type": "classification",
        "hyper_tuning": "random",
        "hyper_budget": 50,
        "stratify_by_groups": "N",
        "groups": "",
        "balancing": "NONE",
        "fit_scorer": "f1_score",
        "scorer_list": ["accuracy_score", "f1_score"],
        "model_list": [
            "RandomForestClassifier",
            "AdaBoostClassifier",
            "KNeighborsClassifier",
            "AutoXGBoost",
            "AutoLGBM",
            "AutoKeras",
        ],
        "autokeras_config": {
            "n_epochs": 100,
            "batch_size": 32,
            "verbose": True,
            "n_blocks": 3,
            "dropout": 0.3,
            "use_batchnorm": True,
            "n_trials": 4,
            "tuner": "bayesian",
        },
        "autolgbm_config": {"verbose": True, "n_trials": 5, "timeout": 60},
        "autoxgboost_config": {"verbose": True, "n_trials": 10, "timeout": 500},
        "feature_selection": {
            "k": 1000,
            "var_threshold": 0,
            "auto": {
                "min_features": 10,
                "interval": 1,
                "eval_model": "RandomForestClassifier",
                "eval_metric": "f1_score",
            },
            "method": {"name": "SelectKBest", "metric": "f_classif"},
        },
        "encoding": None,
    },
    "plotting": {
        "plot_method": [
            "barplot_scorer",
            "boxplot_scorer",
            "conf_matrix",
            "shap_plots",
            "roc_curve",
        ],
        "top_feats_permImp": 15,
        "top_feats_shap": 15,
        "explanations_data": "all",
    },
    "tabular": {
        "filter_tabular_sample": 1000000,
        "filter_tabular_measurements": [0, 1],
    },
}


class Test_Model:
    def test_testConfig(self):
        try:
            Model(**TEST_CONFIG)
            assert True
        except Exception:
            assert False

    @pytest.mark.parametrize(
        "key", [k for k, v in Model.model_fields.items() if v.is_required()]
    )
    def test_missing_required(self, key):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG)
        del MODIFIED_CONFIG[key]

        try:
            Model(**MODIFIED_CONFIG)
            assert False
        except Exception as e:
            errs = e.errors()
            assert len(errs) == 1
            errs = errs[0]
            assert errs["type"] == "missing"
            assert errs["loc"][0] == key

    @pytest.mark.parametrize(
        "data_type",
        ["tabular", "gene_expression", "microbiome", "metabolomic", "other"],
    )
    def test_nulling_omic_entry(self, data_type):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG)
        MODIFIED_CONFIG["data"]["data_type"] = data_type
        model = Model(**MODIFIED_CONFIG)

        other_omics = [
            "tabular",
            "gene_expression",
            "microbiome",
            "metabolomic",
        ]

        if data_type != "other":
            assert model.model_dump()[data_type] is not None
            other_omics.remove(data_type)

        for omic in other_omics:
            assert model.model_dump()[omic] is None
