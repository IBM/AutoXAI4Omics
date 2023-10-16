from pathlib import Path
import metrics.metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_val_score,
    RandomizedSearchCV,
    GridSearchCV,
)  # , KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from xgboost import XGBClassifier, XGBRegressor
from metrics.metrics import evaluate_model

import models.model_params as model_params

from models.custom_model import CustomModel
from models.custom_model import (
    FixedKeras,
    AutoKeras,
    AutoSKLearn,
    AutoLGBM,
    AutoXGBoost,
)  # , AutoGluon

import logging
from utils.save import save_results
from utils.save import save_model


import os

from plotting.plots_both import plot_model_performance

##### Fix for each thread spawning its own GUI is to use 1 thread
##### Change this to n_jobs = -1 for all-core processing (when we get that working)
n_jobs = -1
omicLogger = logging.getLogger("OmicLogger")


def select_model_dict(hyper_tuning):
    """
    Select what parameter range specific we are using based on the given hyper_tuning method.
    """
    omicLogger.debug("Get tunning settings...")

    if hyper_tuning == "random":
        ref_model_dict = model_params.sk_random
    elif hyper_tuning == "grid":
        ref_model_dict = model_params.sk_grid
    elif hyper_tuning is None:
        ref_model_dict = model_params.single_model
    else:
        raise ValueError(f"{hyper_tuning} is not a valid option")
    return ref_model_dict


def define_models(problem_type, hyper_tuning):
    """
    Define the models to be run.

    The name is the key, the value is a tuple with the model function, and defined params
    """
    omicLogger.debug("Defining the set of models...")
    ref_model_dict = select_model_dict(hyper_tuning)

    if problem_type == "classification":
        try:
            # Specific modifications for problem type
            if hyper_tuning is None or hyper_tuning == "boaas":
                ref_model_dict["svm"]["probability"] = True
            else:
                ref_model_dict["svm"]["probability"] = [True]
        # Otherwise pass - models may not always be defined for every tuning method
        except KeyError:
            pass
        # Define dict
        model_dict = {
            "rf": (RandomForestClassifier, ref_model_dict["rf"]),
            "svm": (SVC, ref_model_dict["svm"]),
            "knn": (KNeighborsClassifier, ref_model_dict["knn"]),
            "adaboost": (AdaBoostClassifier, ref_model_dict["adaboost"]),
            "xgboost": (XGBClassifier, ref_model_dict["xgboost"]),
        }
    elif problem_type == "regression":
        # Specific modifications for problem type

        # Define dict
        model_dict = {
            "rf": (RandomForestRegressor, ref_model_dict["rf"]),
            "svm": (SVR, ref_model_dict["svm"]),
            "knn": (KNeighborsRegressor, ref_model_dict["knn"]),
            "adaboost": (AdaBoostRegressor, ref_model_dict["adaboost"]),
            "xgboost": (XGBRegressor, ref_model_dict["xgboost"]),
        }
    else:
        raise ValueError(f"{problem_type} is not recognised, must be either 'regression' or 'classification'")
    # The CustomModels handle classification and regression themselves so put outside
    # For mixing tuning types, default to using the single model for mlp_ens
    try:
        model_dict["fixedkeras"] = (FixedKeras, ref_model_dict["fixedkeras"])
        model_dict["autokeras"] = (AutoKeras, ref_model_dict["autokeras"])
        model_dict["autolgbm"] = (AutoLGBM, ref_model_dict["autolgbm"])
        model_dict["autoxgboost"] = (AutoXGBoost, ref_model_dict["autoxgboost"])
        model_dict["autosklearn"] = (AutoSKLearn, ref_model_dict["autosklearn"])
        # model_dict["autogluon"] = (AutoGluon, ref_model_dict['autogluon'])
    except KeyError:
        model_dict["fixedkeras"] = (FixedKeras, model_params.single_model["fixedkeras"])
        model_dict["autokeras"] = (AutoKeras, model_params.single_model["autokeras"])
        model_dict["autolgbm"] = (AutoLGBM, model_params.single_model["autolgbm"])
        model_dict["autoxgboost"] = (
            AutoXGBoost,
            model_params.single_model["autoxgboost"],
        )
        model_dict["autosklearn"] = (
            AutoSKLearn,
            model_params.single_model["autosklearn"],
        )
        # model_dict["autogluon"] = (AutoGluon, model_params.single_model['autogluon'])
    return model_dict


########## EVALUATE ##########
def best_selector(experiment_folder, problem_type, metric=None, collapse_tax=None):
    """
    Give trained models this will find and select the best one
    """

    if collapse_tax is None:
        collapse_tax = ""

    omicLogger.debug("selecting best model...")
    filepath = experiment_folder / f"results/scores_{collapse_tax}_performance_results_testset.csv"

    if not os.path.exists(filepath):
        raise ValueError(f"{filepath} does not exist")

    df = pd.read_csv(filepath)
    df = df.set_index("model")

    if problem_type == "classification":
        if metric is None:
            omicLogger.info("Best selection metric is None, Defaulting to F1_score...")
            metric = "f1"
        low = False
    else:
        if metric is None:
            omicLogger.info("Best selection metric is None, Defaulting to Mean_AE...")
            metric = "mean_ae"
        low = True

    df_cols = list(
        set([x.replace("_Train", "").replace("_Test", "") for x in list(df.columns) if ("PerClass" not in x)])
    )
    offical_name = [x for x in df_cols if (metric in x.lower())]

    if len(offical_name) == 0:
        raise ValueError(f"{metric} not in metrics calculated for models")

    metric = offical_name[0]

    t_df = df[[metric + "_Train", metric + "_Test"]]

    plot_model_performance(experiment_folder, t_df, metric, low=low)

    ang = t_df.apply(
        lambda row: round(
            np.arccos(np.dot(row.values, [1, 1]) / (np.linalg.norm(row.values) * np.linalg.norm([1, 1]))),
            4,
        ),
        axis=1,
    )
    ang.name = "Angle"

    nrm = t_df.apply(lambda row: round(np.linalg.norm(row.values - 1 + int(low)), 4), axis=1)
    nrm.name = "Norm"

    best = pd.concat([nrm, ang], axis=1)
    best.sort_values(by=["Norm", "Angle"], inplace=True)

    nrm_min = list(np.where(best["Norm"] == best["Norm"].min())[0])
    sub1 = best["Angle"].iloc[nrm_min]
    ang_min = list(np.where(sub1 == sub1.min())[0])
    best_models = list(sub1.keys()[ang_min])

    return best_models


########## WRAPPERS ##########
def random_search(
    model,
    model_name,
    param_ranges,
    budget: int,
    x_train,
    y_train,
    seed_num: int,
    scorer_dict,
    fit_scorer: str,
):
    """
    Wrapper for using sklearn's RandomizedSearchCV
    """
    omicLogger.debug("Training with a random search...")
    # If possible, set the random state for the model
    try:
        # Just a dummy to see if the model has a random state attribute
        # Improvement would be if there is hasattr func but for arguments
        _ = model(random_state=0)
        param_ranges["random_state"] = [seed_num]
    except TypeError:
        pass
    # Setup the random search with cross val
    print("Setup the random search with cross val")
    random_search = RandomizedSearchCV(
        estimator=model(),
        param_distributions=param_ranges,
        n_iter=budget,
        cv=5,
        verbose=1,
        n_jobs=n_jobs,
        random_state=seed_num,
        pre_dispatch="2*n_jobs",
        scoring=scorer_dict,
        refit=fit_scorer,
    )

    # Fit the random search
    print("Fit the random search")
    try:
        random_search.fit(x_train, y_train)
    except ValueError:
        print("!!! ERROR - PLEASE SELECT VALID TARGET AND PREDICTION TASK")
        raise
    # Return the best estimator found
    print(random_search.best_estimator_)
    return random_search.best_estimator_


def grid_search(
    model,
    model_name,
    param_ranges,
    x_train,
    y_train,
    seed_num,
    scorer_dict,
    fit_scorer: str,
):
    """
    Wrapper for using sklearn's GridSearchCV
    """
    omicLogger.debug("Training with a grid search...")
    try:
        _ = model(random_state=0)
        param_ranges["random_state"] = [seed_num]
    except TypeError:
        pass

    grid_search = GridSearchCV(
        estimator=model(),
        param_grid=param_ranges,
        cv=5,
        verbose=1,
        n_jobs=n_jobs,
        pre_dispatch="2*n_jobs",
        scoring=scorer_dict,
        refit=fit_scorer,
    )
    # Fit the random search
    grid_search.fit(x_train, y_train)
    # Return the best estimator found
    print(grid_search.best_estimator_)
    return grid_search.best_estimator_


def single_model(model, param_ranges, x_train, y_train, seed_num):
    """
    Wrapper for training and setting up a single model (i.e. no tuning).
    """
    omicLogger.debug("Training as single model...")
    try:
        _ = model(random_state=0)
        param_ranges["random_state"] = seed_num
    except TypeError:
        pass
    print(model().set_params(**param_ranges))
    trained_model = model().set_params(**param_ranges).fit(x_train, y_train)
    return trained_model


########## RUN MODELS ##########
def predict_model(model, x_train, y_train, x_test=None):
    """
    Generic function to fit a model and return predictions on train and test data (if given)
    """
    omicLogger.debug("Predicting with given model...")
    model.fit(x_train, y_train)
    train_preds = model.predict(x_train)
    if x_test is not None:
        test_preds = model.predict(x_test)
    else:
        test_preds = None
    return train_preds, test_preds


def run_models(
    config_dict,
    model_list,
    model_dict,
    df_train,
    df_test,
    x_train,
    y_train,
    x_test,
    y_test,
    experiment_folder,
    fit_scorer,
    hyper_tuning,
    hyper_budget,
    problem_type,
    seed_num,
):
    """
    Run (and tune if applicable) each of the models sequentially, saving the results and models.
    """
    omicLogger.debug("Initialised training & tuning of models...")

    # Construct the filepath to save the results
    results_folder = experiment_folder / "results"

    # Create dataframe for performance results
    df_performance_results = pd.DataFrame()

    fname = "scores_"

    if config_dict["data"]["data_type"] == "microbiome":
        # This is specific to microbiome
        if config_dict["microbiome"]["collapse_tax"] is not None:
            fname += config_dict["microbiome"]["collapse_tax"]
        # Remove or merge samples based on target values (for example merging to categories, if classification)
        if config_dict["microbiome"]["remove_classes"] is not None:
            fname += "_remove"
        elif config_dict["microbiome"]["merge_classes"] is not None:
            fname += "_merge"

    # Just need it here for determing tuning logic
    ref_model_dict = select_model_dict(hyper_tuning)
    # So that we can pass the func to the CustomModels

    #  Define all the scores
    scorer_dict = metrics.metrics.define_scorers(config_dict["ml"]["problem_type"], config_dict["ml"]["scorer_list"])
    scorer_func = scorer_dict[config_dict["ml"]["fit_scorer"]]

    # Run each model
    for model_name in model_list:
        omicLogger.debug(f"Training model: {model_name}")
        print(f"Testing {model_name}")

        # Placeholder variable to handle mixed hyperparam tuning logic for MLPEnsemble
        single_model_flag = False

        # Load the model and it's parameter path
        model, param_ranges = model_dict[model_name]

        # Setup the CustomModels
        if model_name in CustomModel.custom_aliases:
            single_model_flag, param_ranges = model.setup_custom_model(
                config_dict["ml"],
                experiment_folder,
                model_name,
                ref_model_dict,
                param_ranges,
                scorer_func,
                x_test,
                y_test,
            )

        # Random search
        if hyper_tuning == "random" and not single_model_flag:
            print("Using random search")
            # Do a random search to find the best parameters
            trained_model = random_search(
                model,
                model_name,
                param_ranges,
                hyper_budget,
                x_train,
                y_train,
                seed_num,
                scorer_dict,
                fit_scorer,
            )
            print("=================== Best model from random search: " + model_name + " ====================")
            print(trained_model)
            print("==================================================================")

        # No hyperparameter tuning (and/or the MLPEnsemble is to be run once)
        elif hyper_tuning is None or single_model_flag:
            if hyper_budget is not None:
                print(f"Hyperparameter tuning budget ({hyper_budget}) is not used without tuning")
            # No tuning, just use the parameters supplied
            trained_model = single_model(model, param_ranges, x_train, y_train, seed_num)

        # Grid search
        elif hyper_tuning == "grid":
            print("Using grid search")
            if hyper_budget is not None:
                print(f"Hyperparameter tuning budget ({hyper_budget}) is not used in a grid search")
            trained_model = grid_search(
                model,
                model_name,
                param_ranges,
                x_train,
                y_train,
                seed_num,
                scorer_dict,
                fit_scorer,
            )
            print("=================== Best model from grid search: " + model_name + " ====================")
            print(trained_model)
            print("==================================================================")

        # Save the best model found
        save_model(experiment_folder, trained_model, model_name)

        # Evaluate the best model using all the scores and CV
        performance_results_dict, predictions = evaluate_model(
            trained_model,
            config_dict["ml"]["problem_type"],
            x_train,
            y_train,
            x_test,
            y_test,
            scorer_dict,
        )
        predictions.to_csv(results_folder / f"{model_name}_predictions.csv", index=False)

        # Save the results
        df_performance_results, fname_perfResults = save_results(
            results_folder,
            df_performance_results,
            performance_results_dict,
            model_name,
            fname,
            suffix="_performance_results_testset",
            save_pkl=False,
            save_csv=True,
        )

        print(f"{model_name} complete! Results saved at {Path(fname_perfResults).parents[0]}")
