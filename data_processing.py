# import subprocess
import utils
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
import scipy.sparse
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
import models
import math
import plotting
import logging

import imblearn

import cProfile
import pstats
import io

###### omic pre-processing
from omics import microbiome, geneExp, metabolomic, tabular

###### FS METHODS
from sklearn.feature_selection import SelectKBest, VarianceThreshold, RFE

######

###### FS METRICS
from sklearn.feature_selection import f_regression, f_classif, mutual_info_regression, mutual_info_classif

######

###### MODEL METRICS
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    hinge_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    zero_one_loss,
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
    mean_poisson_deviance,
    mean_gamma_deviance,
    mean_tweedie_deviance,
)
import inspect

######

###### METRICS TO BE IMPORTED WITH SKL V1.0.2
# mean_absolute_percentage_error, d2_tweedie_score, mean_pinball_loss
######

###### CLASSIFIERS
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

######

###### REGRESSORS
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    RidgeCV,
    SGDRegressor,
    ElasticNet,
    ElasticNetCV,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

######

omicLogger = logging.getLogger("OmicLogger")


################### STANDARDIS DATA ###################
def standardize_data(data):
    """
    Standardize the input X using Standard Scaler
    """
    omicLogger.debug("Applying Standard scaling to given data...")

    if scipy.sparse.issparse(data):
        data = data.todense()
    else:
        data = data

    # SS = StandardScaler()
    SS = QuantileTransformer(n_quantiles=max(20, data.shape[0] // 20), output_distribution="normal")
    data = SS.fit_transform(data)
    return data, SS


def transform_data(data, transformer):
    omicLogger.debug("Transforming given data according to given transformer...")

    if scipy.sparse.issparse(data):
        data = data.todense()
    else:
        data = data

    try:
        data = transformer.transform(data)
        return data
    except:
        raise TypeError("Supplied transformer does not have the transform method")


################### LOAD DATA ###################
def get_data(path_file, target, metadata_path, prediction=False):
    """
    Read the input files and return X, y (target) and the feature_names
    """
    omicLogger.debug("Inserting data into DataFrames...")
    # Read the data
    data = pd.read_csv(path_file)

    # check if the first column is meant to be the index
    # Assumption is that if the first column name is empty/none then it is meant to be an index
    if ("Unnamed" in data.columns[0]) or (data.columns[0] is None):
        data.set_index(data.columns[0], inplace=True)
        data.index.name = None

    print("Data dimension: " + str(data.shape))

    if not prediction:
        # Check if the target is in a separate file or in the same data
        if metadata_path == "":
            y = data[target].values
            data_notarget = data.drop(target, axis=1)

        else:  # it assumes the data does not contain the target column
            # Read the metadata file
            metadata = pd.read_csv(metadata_path, index_col=0)
            y = metadata[target].values
            data_notarget = data

        features_names = data_notarget.columns
        x = data_notarget  # .values

        # Check the data and labels are the right size
        assert len(x) == len(y)

    else:
        try:
            data_notarget = data.drop(target, axis=1)
        except:
            data_notarget = data

        x = data_notarget
        features_names = data_notarget.columns
        y = None

    return x, y, features_names


def load_data(config_dict, load_holdout=False, load_prediction=False):
    """
    A function to handel all of the loading of the data presented in the config file

    Parameters
    ----------
    load_holdout : bool or None
        A var used to control which data to be loaded if: none, returns only the holdout data; True, returns both the non-holdout and the holdout data; False, returns only the non-holdout data.
    load_prediction : bool
        If true will not load the main or hold out datasets but the data to be predicted

    Returns
    -------
    x : Numpy array
        if load_prediction was true then this is the data to be predicted on other wise it is the data to train/test on
    y : Numpy array
        if load_prediction was true then this is not returned on other wise it is the y values for the train/test data
    features_names : list of str
        The list of feature names.
    x_heldout : Numpy array
        if load_prediction was true and load_holdout was None/True then this is the heldout data
    y_heldout : Numpy array
        if load_prediction was true and load_holdout was None/True then this is the y values for the heldout data
    """
    omicLogger.debug("Data load inititalised")

    if not load_prediction:
        if load_holdout is not None:
            x, y, features_names = load_data_main(config_dict)

        if (load_holdout is None) or load_holdout:
            x_heldout, y_heldout, features_names = load_data_holdout(config_dict)

        omicLogger.debug("Load completed")
        if load_holdout is None:
            return x_heldout, y_heldout, features_names
        elif load_holdout:
            return x, y, x_heldout, y_heldout, features_names
        else:
            return x, y, features_names
    else:
        x, features_names = load_data_prediction(config_dict)
        return x, features_names


def load_data_main(config_dict):
    omicLogger.debug("Loading training data...")

    if config_dict["data"]["data_type"] == "microbiome":
        # This reads and preprocesses microbiome data using calour library -- it would be better to change this preprocessing so that it is not dependent from calour
        x, y, features_names = microbiome.get_data_microbiome(
            config_dict["data"]["file_path"], config_dict["data"]["metadata_file"], config_dict
        )
    elif config_dict["data"]["data_type"] == "gene_expression":
        # This reads and preprocesses microbiome data using calour library -- it would be better to change this preprocessing so that it is not dependent from calour
        x, y, features_names = geneExp.get_data_gene_expression(config_dict)
    elif config_dict["data"]["data_type"] == "metabolomic":
        x, y, features_names = metabolomic.get_data_metabolomic(config_dict)
    elif config_dict["data"]["data_type"] == "tabular":
        x, y, features_names = tabular.get_data_tabular(config_dict)
    else:
        # At the moment for all the other data types, for example metabolomics, we have not implemented preprocessing except for standardisation with StandardScaler()
        x, y, features_names = get_data(
            config_dict["data"]["file_path"], config_dict["data"]["target"], config_dict["data"]["metadata_file"]
        )

    return x, y, features_names


def load_data_holdout(config_dict):
    omicLogger.debug("Training loaded. Loading holdout data...")
    if config_dict["data"]["data_type"] == "microbiome":
        # This reads and preprocesses microbiome data using calour library -- it would be better to change this preprocessing so that it is not dependent from calour
        x_heldout, y_heldout, features_names = microbiome.get_data_microbiome(
            config_dict["data"]["file_path_holdout_data"],
            config_dict["data"]["metadata_file_holdout_data"],
            config_dict,
        )
    elif config_dict["data"]["data_type"] == "gene_expression":
        # This reads and preprocesses microbiome data using calour library -- it would be better to change this preprocessing so that it is not dependent from calour
        x_heldout, y_heldout, features_names = geneExp.get_data_gene_expression(config_dict, holdout=True)
    elif config_dict["data"]["data_type"] == "metabolomic":
        x_heldout, y_heldout, features_names = metabolomic.get_data_metabolomic(config_dict, holdout=True)
    elif config_dict["data"]["data_type"] == "tabular":
        x_heldout, y_heldout, features_names = tabular.get_data_tabular(config_dict, holdout=True)
    else:
        # At the moment for all the other data types, for example metabolomics, we have not implemented preprocessing except for standardisation with StandardScaler()
        x_heldout, y_heldout, features_names = get_data(
            config_dict["data"]["file_path_holdout_data"],
            config_dict["data"]["target"],
            config_dict["data"]["metadata_file_holdout_data"],
        )

    return x_heldout, y_heldout, features_names


def load_data_prediction(config_dict):
    """
    TODO - INPROGRESS
    """
    omicLogger.debug("Loading prediction data")

    if config_dict["data"]["data_type"] == "microbiome":
        x, _, features_names = microbiome.get_data_microbiome_trained(config_dict, holdout=False, prediction=True)

    elif config_dict["data"]["data_type"] == "gene_expression":
        x, _, features_names = geneExp.get_data_gene_expression_trained(config_dict, holdout=False, prediction=True)

    elif config_dict["data"]["data_type"] == "metabolomic":
        x, _, features_names = metabolomic.get_data_metabolomic_trained(config_dict, holdout=False, prediction=True)

    elif config_dict["data"]["data_type"] == "tabular":
        x, _, features_names = tabular.get_data_tabular_trained(config_dict, holdout=False, prediction=True)

    else:
        # TODO should work for prediction for now, but if not edit
        x, _, features_names = get_data(config_dict["prediction"]["file_path"], config_dict["data"]["target"], "", True)

    return x, features_names


################### SPLIT DATA ###################


def split_data(x, y, config_dict):
    """
    Split the data according to the config (i.e normal split or stratify by groups)
    """

    omicLogger.debug("Splitting data...")
    # Split the data in train and test
    if config_dict["ml"]["stratify_by_groups"] == "Y":
        x_train, x_test, y_train, y_test = strat_split(x, y, config_dict)

    else:
        x_train, x_test, y_train, y_test = std_split(x, y, config_dict)

    return x_train, x_test, y_train, y_test


def strat_split(x, y, config_dict):
    """
    split the data according to stratification
    """
    omicLogger.debug("Splitting according to stratification...")

    gss = GroupShuffleSplit(
        n_splits=1, test_size=config_dict["ml"]["test_size"], random_state=config_dict["ml"]["seed_num"]
    )
    # gss = GroupKFold(n_splits=7)

    metadata = pd.read_csv(config_dict["data"]["metadata_file"], index_col=0)
    le = LabelEncoder()
    groups = le.fit_transform(metadata[config_dict["ml"]["groups"]])

    for train_idx, test_idx in gss.split(x, y, groups):
        x_train, x_test, y_train, y_test = x[train_idx], x[test_idx], y[train_idx], y[test_idx]

    return x_train, x_test, y_train, y_test


def std_split(x, y, config_dict):
    """
    Determine the type of train test split to use on the data.
    """
    omicLogger.debug("Split according to standard methods...")

    test_size = config_dict["ml"]["test_size"]
    seed_num = config_dict["ml"]["seed_num"]
    problem_type = config_dict["ml"]["problem_type"]

    if problem_type == "classification":
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, random_state=seed_num, stratify=y
            )
        except:
            print("!!! ERROR: PLEASE SELECT VALID PREDICTION TASK AND TARGET !!!")
            raise

    # Don't stratify for regression (sklearn can't currently handle it with e.g. binning)
    elif problem_type == "regression":
        try:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed_num)
        except:
            print("!!! ERROR: PLEASE SELECT VALID PREDICTION TASK AND TARGET !!!")
            raise

    return x_train, x_test, y_train, y_test


################### FEATURE SELECTION ###################
# Potential improvements:
# ability to choose feature selection method --- DONE (TODO: LOAD IN OTHER METHODS WHEN SKLEARN VERSION GETS UPDATED)
# ability to choose selection model used --- DONE (TODO: LOAD IN EXTRA CLASSIFIERS & TEST)
# ability to choose selection model evaluation metric --- DONE (TODO: TEST ALL METRICS & LOAD IN EXTRAS WHEN SKLEARN GETS UPDATED)


def manual_feat_selection(x, y, k_select, problem_type, method_dict):
    """
    Given trainging data this will select the k best features for predicting the target. we assume data has been split into test-train and standardised
    """
    omicLogger.debug(f"Selecting {k_select} features...")
    if method_dict["name"] == "SelectKBest":
        metric = globals()[method_dict["metric"]]
        fs_method = SelectKBest(metric, k=k_select)
    elif method_dict["name"] == "RFE":
        estimator = globals()[method_dict["estimator"]](random_state=42, n_jobs=-1)
        fs_method = RFE(estimator, n_features_to_select=k_select, step=1)
    else:
        raise ValueError(f"{method_dict['name']} is not available for use, please select another method.")

    # perform selection
    x_trans = fs_method.fit_transform(x, y)

    return x_trans, fs_method


def train_eval_feat_selection_model(x, y, n_feature, problem_type, eval_model=None, eval_metric=None, method_dict=None):
    """
    Train and score a model if it were to only use n_feature
    """
    omicLogger.debug("Selecting features, training model and evaluating for given K...")

    x_trans, SKB = manual_feat_selection(x, y, n_feature, problem_type, method_dict)  # select the best k features

    # check the combination of model and metric is valid
    # eval_model, eval_metric, _ = parse_model_inputs(problem_type, eval_model, eval_metric)

    # init the model and metric functions
    selection_model = globals()[eval_model]
    metric = globals()[eval_metric]

    # init the model
    fs_model = selection_model(n_jobs=-1, random_state=42, verbose=0, warm_start=False)

    # fit, predict, score
    fs_model.fit(x_trans, y)
    y_pred = fs_model.predict(x_trans)
    kwa = {}
    # if 'pos_label' in inspect.signature(metric).parameters.keys():
    #     kwa['pos_label'] = sorted(list(set(y)))[0]

    if "average" in inspect.signature(metric).parameters.keys():
        kwa["average"] = "weighted"
    eval_score = metric(y, y_pred, **kwa)

    return eval_score


def k_selector(experiment_folder, acc, top=True, low=True, save=True):
    """
    Given a set of accuracy results will choose the lowest scoring, stable k
    """
    omicLogger.debug("Selecting optimium K...")

    acc = dict(sorted(acc.items()))  # ensure keys are sorted low-high
    sr = pd.DataFrame(pd.Series(acc))  # turn results dict into a series

    sr["r_m"] = sr[0].rolling(3, center=True).mean()  # compute the rolling average, based on a window of 3
    sr["r_std"] = sr[0].rolling(3, center=True).std()  # compute the rolling std, based on a window of 3

    sr["r_m"].iloc[0] = sr[0].iloc[[0, 1]].mean()  # fill in the start values
    sr["r_std"].iloc[0] = sr[0].iloc[[0, 1]].std()

    sr["r_m"].iloc[-1] = sr[0].iloc[[-2, -1]].mean()  # fill in the end values
    sr["r_std"].iloc[-1] = sr[0].iloc[[-2, -1]].std()

    if all(sr[["r_m", "r_std"]].std() == 0):
        print("CONSISTENT RESULTS - picking lightweight model")
        k_selected = sr.index[0]
    else:
        sr_n = (sr[["r_m", "r_std"]] - sr[["r_m", "r_std"]].mean()) / sr[["r_m", "r_std"]].std()  # normalise the values

        plotting.opt_k_plot(experiment_folder, sr_n, save)

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
    interval=1,
    eval_model=None,
    eval_metric=None,
    low=None,
    method_dict=None,
    save=True,
):
    """
    Given data this will automatically find the best number of features, we assume the data provided has already been split into test-train and standardised.
    """
    omicLogger.debug("Initialising the automated selection process...")

    print("Generating logarithmic selection for k")
    max_features = x.shape[1]
    # if the max number of features is infact smaller than min_features, set min_features to 1
    if max_features - 3 <= min_features:
        print("Min features more than given number of features, setting min_features to 1.")
        min_features = 1

    # get a logarithmic spaced selction of potential k
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

    # parse the model evaluation settings
    # eval_model, eval_metric, low = parse_model_inputs(problem_type, eval_model, eval_metric)

    # init result dict
    acc = {}

    # train and evaluate a model for each potential k
    for n_feature in n_feature_candicates:
        print(f"Evaluating basic model trained on {n_feature} features")
        acc[n_feature] = train_eval_feat_selection_model(
            x, y, n_feature, problem_type, eval_model, eval_metric, method_dict
        )

    # plot feat-acc
    plotting.feat_acc_plot(experiment_folder, acc, save)

    print("Selecting optimum k")
    chosen_k = k_selector(
        experiment_folder, acc, low=low, save=save
    )  # select the 'best' k based on the results we have attained

    print("transforming data based on optimum k")
    # get the transformed dataset and the transformer
    x_trans, SKB = manual_feat_selection(x, y, chosen_k, problem_type, method_dict)

    return x_trans, SKB


def feat_selection(experiment_folder, x, y, features_names, problem_type, FS_dict, save=True):
    """
    A function to activate manual or auto feature selection
    """
    omicLogger.debug("Initalising feature selection process...")

    # extract out parameters from the feature selection dict

    k, threshold, method_dict, auto_dict = parse_FS_settings(problem_type, FS_dict)

    # apply variance threholding
    if threshold >= 0:
        print("Applying varience threshold")
        x_trans, VT = variance_removal(x, threshold)
    else:
        raise ValueError("Threshold must be greater than or equal to 0")

    # begin either the automated FS process or the manual one
    if k == "auto":
        print("Beginning Automatic feature selection")
        x_trans, SKB = auto_feat_selection(
            experiment_folder, x_trans, y, problem_type, **auto_dict, method_dict=method_dict, save=save
        )
    elif isinstance(k, int):
        print("Beginning feature selection with given k")
        x_trans, SKB = manual_feat_selection(x_trans, y, k, problem_type, method_dict)
    else:
        raise ValueError("k must either be an int or the string 'auto' ")

    # construct the pipline of transformers
    union = Pipeline([("variance", VT), ("featureSeletor", SKB)])

    # fetch the name of remaining features after the FS pipeline
    feature_names_out = features_names[VT.get_support(indices=True)][SKB.get_support(indices=True)]

    return x_trans, feature_names_out, union


def variance_removal(x, threshold=0):
    omicLogger.debug("Applying variance thresholding...")
    selector = VarianceThreshold(threshold)
    x_trans = selector.fit_transform(x)

    return x_trans, selector


def validate_models_and_metrics(problem_type, estimator, metric):
    """
    Check if given the problem type that the estimator and evaluation metric chosen is valid or not
    """
    omicLogger.debug("Validating model and metric settings...")
    # check that the estimator is loaded in
    if estimator not in globals():
        raise ValueError(f"{estimator} is not currently available for use")
    else:
        est = globals()[estimator]

    # check that the metric is loaded in
    if metric not in globals():
        raise ValueError(f"{metric} is not currently available for use")

    # check that the estimator selected is appropriate for the problem type
    if not (
        ((problem_type == "regression") and (est._estimator_type == "regressor"))
        or ((problem_type == "classification") and (est._estimator_type == "classifier"))
    ):
        raise ValueError(f"{estimator} is not a valid method for a {problem_type} problem")

    # LIMITATION --- for now only using metrics that take y_true and y_predict
    metrics_classifiers = [
        "accuracy_score",
        "f1_score",
        "hamming_loss",
        "hinge_loss",
        "jaccard_score",
        "log_loss",
        "matthews_corrcoef",
        "precision_score",
        "recall_score",
        "zero_one_loss",
    ]
    metrics_regressors = [
        "explained_variance_score",
        "mean_absolute_error",
        "mean_squared_error",
        "mean_squared_log_error",
        "median_absolute_error",
        "r2_score",
        "mean_poisson_deviance",
        "mean_gamma_deviance",
        "mean_tweedie_deviance",
    ]  #'mean_absolute_percentage_error', 'd2_tweedie_score', 'mean_pinball_loss']

    # check that the metric selected is appropriate for the problem types
    if not (
        ((problem_type == "regression") and (metric in metrics_regressors))
        or ((problem_type == "classification") and (metric in metrics_classifiers))
    ):
        raise ValueError(f"{metric} is not a valid method for a {problem_type} problem")

    return utils.low_metric_objective(metric)


def parse_model_inputs(problem_type, eval_model, eval_metric):
    omicLogger.debug("Parsing model inputs...")
    # check we have a valid problem type
    if not ((problem_type == "classification") or (problem_type == "regression")):
        raise ValueError("PROBLEM TYPE IS NOT CLASSIFICATION OR REGRESSION")

    # set the evaluation model and metric if we have been given None
    if eval_model is None:
        eval_model = "RandomForestClassifier" if problem_type == "classification" else "RandomForestRegressor"

    if eval_metric is None:
        eval_metric = "f1_score" if problem_type == "classification" else "mean_squared_error"

    # check the combination of model and metric is valid
    low = validate_models_and_metrics(problem_type, eval_model, eval_metric)

    return eval_model, eval_metric, low


def parse_FS_settings(problem_type, FS_dict):
    """
    A function to check ALL the FS setting to ensure correct/valid entiries/combinations
    """
    omicLogger.debug("Parsing feature selection settings...")

    keys = FS_dict.keys()

    if "k" in keys:
        k = FS_dict["k"]
    else:
        k = "auto"

    if "var_threshold" in keys:
        threshold = FS_dict["var_threshold"]
    else:
        threshold = 0

    if "method" in keys:
        method_dict = FS_dict["method"]

        if method_dict["name"] not in globals():
            raise ValueError(
                f"{method_dict['name']} not currently available for use. please select a different method."
            )

        elif method_dict["name"] == "SelectKBest":
            if ("metric" not in method_dict.keys()) or (method_dict["metric"] is None):
                method_dict["metric"] = "f_classif" if problem_type == "classification" else "f_regression"

            elif method_dict["metric"] not in globals():
                raise ValueError(
                    f"{method_dict['metric']} not currently available for use. please select a different metric."
                )

            else:
                metrics_reg = ["f_regression", "mutual_info_regression"]
                metrics_clf = ["f_classif", "mutual_info_classif"]

                if ((problem_type == "classification") and (method_dict["metric"] not in metrics_clf)) or (
                    (problem_type == "regression") and (method_dict["metric"] not in metrics_reg)
                ):
                    raise ValueError(f"{method_dict['metric']} is not appropriate for problem type {problem_type}.")

        elif method_dict["name"] == "RFE":
            if ("estimator" not in method_dict.keys()) or (method_dict["estimator"] is None):
                method_dict["estimator"] = (
                    "RandomForestClassifier" if problem_type == "classification" else "RandomForestRegressor"
                )
            elif method_dict["estimator"] not in globals():
                raise ValueError(
                    f"{method_dict['estimator']} not currently available for use. please select a different estimator."
                )
            else:
                if (
                    (globals()[method_dict["estimator"]]._estimator_type == "regressor")
                    and (problem_type == "classification")
                ) or (
                    (globals()[method_dict["estimator"]]._estimator_type == "classifier")
                    and (problem_type == "regression")
                ):
                    raise ValueError(f"{method_dict['estimator']} is not appropriate for problem type {problem_type}.")
    else:
        method_dict = {
            "name": "SelectKBest",
            "metric": "f_classif" if problem_type == "classification" else "f_regression",
        }

    if "auto" in keys:
        auto_dict = FS_dict["auto"]
        if "min_features" not in auto_dict.keys():
            auto_dict["min_features"] = 10

        if "interval" not in auto_dict.keys():
            auto_dict["interval"] = 1

        if "eval_model" not in auto_dict.keys():
            auto_dict["eval_model"] = None

        if "eval_metric" not in auto_dict.keys():
            auto_dict["eval_metric"] = None

        auto_dict["eval_model"], auto_dict["eval_metric"], auto_dict["low"] = parse_model_inputs(
            problem_type, auto_dict["eval_model"], auto_dict["eval_metric"]
        )
    else:
        auto_dict = {
            "min_features": 10,
            "interval": 1,
            "eval_model": "RandomForestClassifier" if problem_type == "classification" else "RandomForestRegressor",
            "eval_metric": "f1_score" if problem_type == "classification" else "mean_squared_error",
            "low": problem_type != "classification",
        }

    if method_dict["name"] == "RFE":
        auto_dict["eval_model"] = method_dict["estimator"]

    return k, threshold, method_dict, auto_dict


################### CLASS BALANCING ###################


def oversample_data(x_train, y_train, seed):
    """
    Given the training set it has a class imbalance problem, this will over sample the training data to balance out the classes
    """
    omicLogger.debug("Oversampling data...")
    # define oversampling strategy
    oversample = imblearn.over_sampling.RandomOverSampler(random_state=seed, sampling_strategy="not majority")
    # fit and apply the transform
    x_resampled, y_resampled = oversample.fit_resample(x_train, y_train)
    print(f"X train data after oversampling shape: {x_train.shape}")
    print(f"y train data after oversampling shape: {y_train.shape}")

    return x_resampled, y_resampled, oversample.sample_indices_


def undersample_data(x_train, y_train, seed):
    """
    Given the training set it has a class imbalance problem, this will over sample the training data to balance out the classes
    """
    omicLogger.debug("Undersampling data...")
    # define undersampling strategy
    oversample = imblearn.under_sampling.RandomUnderSampler(random_state=seed, sampling_strategy="not minority")
    # fit and apply the transform
    x_resampled, y_resampled = oversample.fit_resample(x_train, y_train)
    print(f"X train data after undersampling shape: {x_train.shape}")
    print(f"y train data after undersampling shape: {y_train.shape}")

    return x_resampled, y_resampled, oversample.sample_indices_


# def combisample_data(x_train, y_train,seed):
#     """
#     Given the training set it has a class imbalance problem, this will over sample the training data to balance out the classes
#     """
#     omicLogger.debug("Combined sampling data...")
#     # define combinedsampling strategy
#     oversample = imblearn.combine.SMOTEENN(random_state=seed)
#     # fit and apply the transform
#     x_resampled, y_resampled = oversample.fit_resample(x_train, y_train)
#     print(f"X train data after combined sampling shape: {x_train.shape}")
#     print(f"y train data after combined sampling shape: {y_train.shape}")

#     return x_resampled, y_resampled, oversample.sample_indices_


################### TIME PROFILE ###################
def prof_to_csv(prof: cProfile.Profile):
    out_stream = io.StringIO()
    pstats.Stats(prof, stream=out_stream).print_stats()
    result = out_stream.getvalue()
    # chop off header lines
    result = "ncalls" + result.split("ncalls")[-1]
    lines = [",".join(line.rstrip().split(None, 5)) for line in result.split("\n")]
    return "\n".join(lines)
