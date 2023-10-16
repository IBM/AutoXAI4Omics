# --------------------------------------------------------------------------
# Licensed Materials - Property of IBM
#
# (C) Copyright IBM Corp. 2019, 2020
# --------------------------------------------------------------------------

import numpy as np
import scipy.stats as sp

sk_random = {
    "rf": {
        "n_estimators": sp.randint(20, 200),
        "max_features": ["auto", "sqrt"],
        "max_depth": sp.randint(10, 70),
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    },
    "svm": {
        "C": 10.0 ** np.arange(-4, 8),
        "gamma": 10.0 ** np.arange(-5, 5),
        "kernel": ["linear", "rbf"],
    },
    "knn": {"n_neighbors": sp.randint(2, 20), "metric": ["euclidean", "manhattan"]},
    "adaboost": {"n_estimators": sp.randint(10, 200)},
    "xgboost": {
        "max_depth": sp.randint(2, 8),
        "learning_rate": np.arange(0.05, 0.91, 0.05),
        "n_estimators": sp.randint(50, 500),
    },
    # "mlp_keras": {
    #     "n_epochs": sp.randint(10, 50),
    #     "batch_size": [20],
    #     "lr": sp.uniform(0.001, 0.049),
    #     "verbose": [False],
    #     "n_blocks": sp.randint(2,6),
    #     "dropout": sp.uniform(0.1, 0.8)
    # }
}

sk_grid = {
    "rf": {
        "n_estimators": range(50, 201, 50),
        "max_features": ["auto", "sqrt"],
        "max_depth": range(10, 71, 10),
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    },
    "svm": {
        "C": 10.0 ** np.arange(-3, 8),
        "gamma": 10.0 ** np.arange(-5, 4),
        "kernel": ["linear", "rbf"],
    },
    "knn": {"n_neighbors": range(1, 21, 2), "metric": ["euclidean", "manhattan"]},
    "adaboost": {"n_estimators": range(50, 201, 50)},
    "xgboost": {
        "max_depth": range(2, 9, 2),
        "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "n_estimators": [100],
    },
    # "mlp_ens": {
    #     "n_estimators": range(10, 51, 10),
    #     "n_epochs": range(5, 21, 3),
    #     "batch_size": [10],
    #     "lr": [0.001, 0.005, 0.01, 0.05],
    #     "layer_sizes": [15]
    # }
}

single_model = {
    "rf": {
        "n_estimators": 100,
        "max_features": "auto",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True,
    },
    "svm": {"C": 1.0, "gamma": "scale", "kernel": "rbf"},
    "knn": {"n_neighbors": 5, "metric": "minkowski"},
    "gradboost": {"learning_rate": 0.1, "n_estimators": 100},
    "adaboost": {"learning_rate": 1.0, "n_estimators": 50},
    "xgboost": {"max_depth": 3, "learning_rate": 0.10, "n_estimators": 100},
    "mlp_keras": {
        "n_epochs": 200,
        "batch_size": 80,
        "lr": 0.001,
        "layer_dict": None,
        "verbose": True,
        "n_blocks": 4,
        "dropout": 0.4,
    },
    "fixedkeras": {
        "n_epochs": 10,
        "batch_size": 80,
        "lr": 0.001,
        "layer_dict": None,
        "verbose": True,
        "n_blocks": 4,
        "dropout": 0.4,
    },
    "autokeras": {},
    "autolgbm": {},
    "autoxgboost": {},
    "autosklearn": {},
}


boaas_dict = {
    "rf": {
        "domain": [
            {"name": "n_estimators", "min": 50, "max": 500, "step": 10},
            {"name": "max_depth", "min": 5, "max": 100, "step": 5},
            {"name": "min_samples_split", "min": 5, "max": 100, "step": 5},
            {"name": "min_samples_leaf", "min": 5, "max": 100, "step": 5},
        ],
        "bootstrap": True,
        "max_features": "auto",
    },
    "xgboost": {
        "domain": [
            {"name": "n_estimators", "min": 50, "max": 500, "step": 10},
            {"name": "max_depth", "min": 2, "max": 20, "step": 1},
            {"name": "learning_rate", "min": 0.05, "max": 0.95, "step": 0.05},
        ]
    },
    "mlp_ens": {
        "domain": [
            {"name": "n_estimators", "min": 10, "max": 100, "step": 5},
            {"name": "n_epochs", "min": 5, "max": 30, "step": 5},
            {"name": "lr", "min": 0.001, "max": 0.05, "step": 0.005},
            {"name": "layer_sizes", "min": 5, "max": 100, "step": 5},
        ],
        "batch_size": 10,
    },
    "mlp_keras": {
        "domain": [
            {"name": "n_epochs", "min": 5, "max": 50, "step": 5},
            {"name": "lr", "min": 0.001, "max": 0.05, "step": 0.005},
            {"name": "n_blocks", "min": 2, "max": 8, "step": 1},
            {"name": "dropout", "min": 0.1, "max": 0.8, "step": 0.05},
        ],
        "batch_size": 20,
    },
    "knn": {
        "domain": [{"name": "n_neighbours", "min": 2, "max": 50, "step": 1}],
        "metric": "euclidean",  # To make this searchable you need a container to allow categorical (integer) selection
    },
    "adaboost": {"domain": [{"name": "n_estimators", "min": 10, "max": 500, "step": 10}]},
    "svm": {}  # This cannot be defined adequately until boaas accepts non-uniform ranges
    # See https://github.ibm.com/machine-learning-daresbury/boaas/issues/13
}
