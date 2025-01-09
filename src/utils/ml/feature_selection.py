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

from sklearn.pipeline import Pipeline
from models.model_defs import MODELS
import plotting.plots_both
import numpy as np
import pandas as pd
import math
from metrics.metric_defs import METRICS
from sklearn.feature_selection import VarianceThreshold
from typing import Union

import logging

from utils.ml.feature_selection_defs import FS_KBEST_METRICS, FS_METHODS

omicLogger = logging.getLogger("OmicLogger")


def variance_removal(x, threshold=0):
    omicLogger.debug(
        f"Applying varience threshold, using {threshold} as the threshold..."
    )
    selector = VarianceThreshold(threshold)
    x_trans = selector.fit_transform(x)

    return x_trans, selector


def manual_feat_selection(x, y, k_select, method_dict, problem_type):
    """
    Given trainging data this will select the k best features for predicting the target. we assume data has been split
    into test-train and standardised
    """
    omicLogger.debug(f"Selecting {k_select} features...")
    if method_dict["name"] == "SelectKBest":
        omicLogger.debug(f"Using {method_dict['name']} with {method_dict['metric']}...")
        metric = FS_KBEST_METRICS[method_dict["metric"]]
        fs_method = FS_METHODS[method_dict["name"]](metric, k=k_select)

    elif method_dict["name"] == "RFE":
        omicLogger.debug(
            f"Using {method_dict['name']} with {method_dict['estimator']}..."
        )
        estimator = MODELS[problem_type][method_dict["estimator"]]["model"](
            random_state=42, n_jobs=-1
        )
        fs_method = FS_METHODS[method_dict["name"]](
            estimator, n_features_to_select=k_select, step=1
        )
    else:
        raise ValueError(
            f"{method_dict['name']} is not available for use, please select another method."
        )

    # perform selection
    x_trans = fs_method.fit_transform(x, y)

    return x_trans, fs_method


def train_eval_feat_selection_model(
    x,
    y_true,
    n_feature: int,
    problem_type: str,
    eval_model: str = None,
    eval_metric: str = None,
    method_dict: dict = None,
):
    """
    Train and score a model if it were to only use n_feature
    """
    omicLogger.debug("Selecting features, training model and evaluating for given K...")

    # select the best k features
    x_trans, SKB = manual_feat_selection(
        x, y_true, n_feature, method_dict, problem_type
    )

    # init the model and metric functions
    omicLogger.debug(f"Init model {eval_model} and metric {eval_metric}")
    selection_model = MODELS[problem_type][eval_model]["model"]
    metric = METRICS[problem_type][eval_metric]

    # init the model
    fs_model = selection_model(n_jobs=-1, random_state=42, verbose=0, warm_start=False)

    # fit, predict, score
    omicLogger.debug("Fitting and predicting with eval model...")
    fs_model.fit(x_trans, y_true)
    y_pred = fs_model.predict(x_trans)
    omicLogger.debug("scoring eval model with eval metic...")
    eval_score = metric._score_func(y_true, y_pred, **metric._kwargs)

    return eval_score


def k_selector(experiment_folder, acc, top=True, low=True, save=True):
    """
    Given a set of accuracy results will choose the lowest scoring, stable k
    """
    omicLogger.debug("Selecting optimium K...")

    acc = dict(sorted(acc.items()))  # ensure keys are sorted low-high
    sr = pd.DataFrame(pd.Series(acc))  # turn results dict into a series

    sr["r_m"] = (
        sr[0].rolling(3, center=True).mean()
    )  # compute the rolling average, based on a window of 3
    sr["r_std"] = (
        sr[0].rolling(3, center=True).std()
    )  # compute the rolling std, based on a window of 3

    sr["r_m"].iloc[0] = sr[0].iloc[[0, 1]].mean()  # fill in the start values
    sr["r_std"].iloc[0] = sr[0].iloc[[0, 1]].std()

    sr["r_m"].iloc[-1] = sr[0].iloc[[-2, -1]].mean()  # fill in the end values
    sr["r_std"].iloc[-1] = sr[0].iloc[[-2, -1]].std()

    if all(sr[["r_m", "r_std"]].std() == 0):
        print("CONSISTENT RESULTS - picking lightweight model")
        k_selected = sr.index[0]
    else:
        sr_n = (sr[["r_m", "r_std"]] - sr[["r_m", "r_std"]].mean()) / sr[
            ["r_m", "r_std"]
        ].std()  # normalise the values

        plotting.plots_both.opt_k_plot(experiment_folder, sr_n, save)

        if low:
            sr_n = sr_n - math.floor(
                sr_n.min().min()
            )  # adjust the values as we are look for the lowest left hand quadrant
        else:
            adj = max(sr_n["r_m"].max(), abs(sr_n["r_std"].min()))
            sr_n["r_m"] = sr_n["r_m"] - adj
            sr_n["r_std"] = sr_n["r_std"] + adj

        sr_r = (sr_n["r_m"] ** 2 + sr_n["r_std"] ** 2) ** 0.5  # compute the norms

        ind_selected = np.where(sr_r == sr_r.min())[
            0
        ].max()  # select the smallest norm, if there are multiple select that with the largest number of features.

        # averages were calculated from a window if top is true it will fetch the top k in this winning window
        if top:
            ind_selected += 1 if ind_selected < len(sr_r.index) - 1 else 0

        # get k value
        k_selected = sr_r.index[ind_selected]

    return k_selected


def auto_feat_selection(
    experiment_folder,
    x,
    y,
    problem_type,
    min_features=10,
    max_features=None,
    interval=1,
    eval_model=None,
    eval_metric=None,
    low=None,
    method_dict=None,
    save=True,
):
    """
    Given data this will automatically find the best number of features, we assume the data provided has already been
    split into test-train and standardised.
    """
    omicLogger.debug("Initialising the automated selection process...")

    n_feature_candicates = generate_k_candicates(
        x, min_features, max_features, interval
    )
    acc = {}

    # train and evaluate a model for each potential k
    for n_feature in n_feature_candicates:
        print(f"Evaluating basic model trained on {n_feature} features")
        acc[n_feature] = train_eval_feat_selection_model(
            x, y, n_feature, problem_type, eval_model, eval_metric, method_dict
        )

    # plot feat-acc
    plotting.plots_both.feat_acc_plot(experiment_folder, acc, save)

    print("Selecting optimum k")
    chosen_k = k_selector(
        experiment_folder, acc, low=low, save=save
    )  # select the 'best' k based on the results we have attained

    print("transforming data based on optimum k")
    # get the transformed dataset and the transformer
    x_trans, SKB = manual_feat_selection(x, y, chosen_k, method_dict, problem_type)

    return x_trans, SKB


def generate_k_candicates(
    x, min_features: int, max_features: int, interval: int
) -> list[int]:
    if max_features is None:
        max_features = x.shape[1]
    elif max_features > x.shape[1]:
        omicLogger.warning(
            f"Max features {max_features} is more than the columns in dataset, defaulting to full number of columns"
        )
        max_features = x.shape[1]
    omicLogger.info(f"Max number of features set to: {max_features}")

    # if the max number of features is infact smaller than min_features, set min_features to 1
    if max_features - 3 <= min_features:
        omicLogger.warn(
            "Min features more than given number of features, setting min_features to 1."
        )
        min_features = 1
    omicLogger.info(f"Min number of features set to: {min_features}")

    # get a logarithmic spaced selction of potential k
    omicLogger.info("Generating logarithmic selection for k...")
    n_feature_candicates = (
        (
            10
            ** np.arange(
                np.log10(min_features),
                np.log10(max_features) // (10**-interval) / (10**interval),
                (10**-interval),
            )
            // 1
        )
        .astype(int)
        .tolist()
    )
    omicLogger.info(f"Selection for k created: {n_feature_candicates}")
    return n_feature_candicates


def feat_selection(
    experiment_folder, x, y, features_names, problem_type, FS_dict, save=True
) -> tuple[Union[pd.DataFrame, np.ndarray], list[str], Pipeline]:
    """
    A function to activate manual or auto feature selection
    """
    omicLogger.debug("Initalising feature selection process...")

    # extract out parameters from the feature selection dict
    k = FS_dict["k"]
    threshold = FS_dict["var_threshold"]
    method_dict = FS_dict["method"]
    auto_dict = FS_dict["auto"]

    # apply variance threholding
    if threshold >= 0:
        x_trans, VT = variance_removal(x, threshold)
    else:
        raise ValueError("Threshold must be greater than or equal to 0")

    # begin either the automated FS process or the manual one
    if k == "auto":
        x_trans, SKB = auto_feat_selection(
            experiment_folder,
            x_trans,
            y,
            problem_type,
            **auto_dict,
            method_dict=method_dict,
            save=save,
        )
    elif isinstance(k, int):
        print("Beginning feature selection with given k")
        x_trans, SKB = manual_feat_selection(x_trans, y, k, method_dict, problem_type)
    else:
        raise ValueError("k must either be an int or the string 'auto' ")

    # construct the pipline of transformers
    union = Pipeline([("variance", VT), ("featureSeletor", SKB)])

    # fetch the name of remaining features after the FS pipeline
    feature_names_out = features_names[VT.get_support(indices=True)][
        SKB.get_support(indices=True)
    ]

    return x_trans, feature_names_out, union
