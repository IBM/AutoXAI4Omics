from pathlib import Path
import metrics.metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from models.model_defs import form_model_dict
from models.custom_model import CustomModel
import logging
from utils.save import save_results
from utils.save import save_model
import os
from plotting.plots_both import plot_model_performance
from utils.vars import CLASSIFICATION

##### Fix for each thread spawning its own GUI is to use 1 thread
##### Change this to n_jobs = -1 for all-core processing (when we get that working)
n_jobs = -1
omicLogger = logging.getLogger("OmicLogger")


########## EVALUATE ##########
def select_best_model(experiment_folder, problem_type, metric=None, collapse_tax=None):
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

    if problem_type == CLASSIFICATION:
        if metric is None:
            omicLogger.info("Best selection metric is None, Defaulting to F1_score...")
            metric = "f1_score"
        low = False
    else:
        if metric is None:
            omicLogger.info("Best selection metric is None, Defaulting to Mean_AE...")
            metric = "mean_absolute_error"
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
    omicLogger.debug(f"Setting parameters: {param_ranges}")
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
    # ref_model_dict = select_model_dict(hyper_tuning)
    # So that we can pass the func to the CustomModels

    #  Define all the scores
    scorer_dict = metrics.metrics.define_scorers(problem_type, config_dict["ml"]["scorer_list"])
    scorer_func = scorer_dict[config_dict["ml"]["fit_scorer"]]

    model_dict = form_model_dict(problem_type, hyper_tuning, model_list)

    # Run each model
    for model_name in model_list:
        omicLogger.debug(f"Training model: {model_name}")
        print(f"Training {model_name}")

        # Placeholder variable to handle mixed hyperparam tuning logic for MLPEnsemble
        # single_model_flag = False

        # Load the model and it's parameter path
        model, param_ranges, single_model_flag = model_dict[model_name]

        # Setup the CustomModels
        # FIXME: I think the following if block can be removed
        if model_name in CustomModel.custom_aliases:
            # single_model_flag, param_ranges = model.setup_custom_model(
            param_ranges = model.setup_custom_model(
                config_dict["ml"],
                experiment_folder,
                model_name,
                # ref_model_dict,
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
        performance_results_dict, predictions = metrics.metrics.evaluate_model(
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
