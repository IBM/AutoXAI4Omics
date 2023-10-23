import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from models.custom_model import CustomModel
from utils.utils import pretty_names
import utils.load
from tensorflow.keras import backend as K
from utils.vars import CLASSIFICATION, REGRESSION

# import utils.utils
from utils.save import save_fig
import seaborn as sns

import matplotlib.pyplot as plt


import glob
import time


import logging

omicLogger = logging.getLogger("OmicLogger")


def plot_model_performance(experiment_folder, data, metric, low, save=True):
    """
    produces a scatter plot of the models and their performance on the training set and test set according to the given
    metric
    """
    omicLogger.debug(f"Creating model performance scatter according to {metric}...")

    ax = sns.scatterplot(
        x=data[metric + "_Train"].tolist(),
        y=data[metric + "_Test"].tolist(),
    )
    ax_min = data.min().min() * 0.75
    ax_max = 1 if not low else data.max().max()
    ax.plot(
        [ax_min, ax_max],
        [ax_min, ax_max],
        "k--",
    )
    ax.set(xlabel="Training set", ylabel="Test set")
    ax.set_title("Model Performance by " + metric)
    for model, row in data.iterrows():
        test = row[metric + "_Test"]
        train = row[metric + "_Train"]
        ax.text(train + 0.02, test, str(model))

    fig = plt.gcf()
    if save:
        plotname = "model_performance_" + metric
        fname = f"{experiment_folder / 'graphs' /plotname }"
        save_fig(fig, fname)
    # Close the figure to ensure we start anew
    plt.clf()
    plt.close()
    # Clear keras and TF sessions/graphs etc.
    K.clear_session()


def opt_k_plot(experiment_folder, sr_n, save=True):
    """
    Produces a scatter plot with each k plotted by their calibrated meand and std
    """
    omicLogger.debug("Creating opt_k_plot...")

    ax = sns.scatterplot(x=sr_n["r_m"].tolist(), y=sr_n["r_std"].tolist(), hue=np.log10(sr_n.index))
    ax.set_title("Performance of various k features")

    m = round(abs(sr_n).max().max() * 1.1, 1)

    ax.set(xlabel="Calibrated mean", ylabel="Calibrated std")
    ax.axvline(0, -m, m)
    ax.axhline(0, -m, m)

    fig = plt.gcf()
    if save:
        fname = f"{experiment_folder / 'graphs' / 'feature_selection_scatter'}"
        save_fig(fig, fname)
    # Close the figure to ensure we start anew
    plt.clf()
    plt.close()
    # Clear keras and TF sessions/graphs etc.
    K.clear_session()


def feat_acc_plot(experiment_folder, acc, save=True):
    """
    Produces a graph showing the number of features vs. the performance of the model when that many features are
    trainined on it
    """
    omicLogger.debug("Creating feat_acc_plot...")

    ax = sns.lineplot(x=list(acc.keys()), y=list(acc.values()), marker="o")
    ax.set_title("Feature selection model accuracy")
    ax.set(xlabel="Number of selected features", ylabel="Model error")
    ax.set(xscale="log")

    fig = plt.gcf()
    if save:
        fname = f"{experiment_folder / 'graphs' / 'feature_selection_accuracy'}"
        save_fig(fig, fname)
    # Close the figure to ensure we start anew
    plt.clf()
    plt.close()
    # Clear keras and TF sessions/graphs etc.
    K.clear_session()


def barplot_scorer(
    experiment_folder,
    model_list: list[str],
    fit_scorer: str,
    scorer_dict,
    data,
    true_labels,
    save=True,
    holdout=False,
):
    """
    Create a barplot for all models in the folder using the fit_scorer from the config.
    """
    omicLogger.debug("Creating barplot_scorer...")
    # Create the plot objects
    fig, ax = plt.subplots()
    # Container for the scores
    all_scores = []
    # Loop over the models
    for model_name in model_list:
        # Load the model

        try:
            model_path = glob.glob(f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}")[0]
        except IndexError as e:
            print("The trained model " + str("*" + model_name + "*.pkl") + " is not present")
            raise e

        print(f"Plotting barplot for {model_name} using {fit_scorer}")
        model = utils.load.load_model(model_name, model_path)
        # Get our single score
        score = np.abs(scorer_dict[fit_scorer](model, data, true_labels))
        all_scores.append(score)
        # Clear keras and TF sessions/graphs etc.
        K.clear_session()
    pretty_model_names = [pretty_names(name, "model") for name in model_list]
    # Make the barplot
    sns.barplot(x=pretty_model_names, y=all_scores, ax=ax)
    # ax.set_xticklabels(pretty_model_names)
    ax.set_ylabel(pretty_names(fit_scorer, "score"))
    ax.set_xlabel("Model")
    ax.set_title("Performance on test data")
    if save:
        fname = f"{experiment_folder / 'graphs' / 'barplot'}_{fit_scorer}"
        fname += "_holdout" if holdout else ""
        save_fig(fig, fname)

    plt.draw()
    plt.tight_layout()
    plt.pause(0.001)
    time.sleep(2)
    # Close the figure to ensure we start anew
    plt.clf()
    plt.close()

    # Clear keras and TF sessions/graphs etc.
    K.clear_session()


def boxplot_scorer_cv_groupby(
    experiment_folder,
    config_dict,
    scorer_dict,
    data,
    true_labels,
    save=True,
    holdout=False,
):
    """
    Create a graph of boxplots for all models in the folder, using the specified fit_scorer from the config.
    """
    omicLogger.debug("Creating boxplot_scorer_cv_groupby...")
    # Create the plot objects
    fig, ax = plt.subplots()
    # Container for the scores
    all_scores = []
    print(f"Size of data for boxplot: {data.shape}")

    # GroupBy Subject ID -- TO FINISH CODING
    metadata = pd.read_csv(config_dict["data"]["metadata_file"], index_col=0)
    le = LabelEncoder()
    groups = le.fit_transform(metadata[config_dict["ml"]["groups"]])

    fold_obj = GroupShuffleSplit(
        n_splits=5,
        test_size=config_dict["ml"]["test_size"],
        random_state=config_dict["ml"]["seed_num"],
    )

    # fold_obj = GroupKFold(n_splits=5)

    # Loop over the models
    for model_name in config_dict["ml"]["model_list"]:
        # Load the model if trained
        try:
            model_path = glob.glob(f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}")[0]
        except IndexError as e:
            print("The trained model " + str("*" + model_name + "*.pkl") + " is not present")
            raise e

        print(
            f"Plotting boxplot for {model_name} using {config_dict['ml']['fit_scorer']} - Grouped By "
            + config_dict["ml"]["groups"]
        )
        # Select the scorer
        scorer_func = scorer_dict[config_dict["ml"]["fit_scorer"]]
        # Container for scores for this cross-val for this model
        scores = []
        num_testsamples_list = []
        num_fold = 0

        for train_idx, test_idx in fold_obj.split(data, true_labels, groups):
            omicLogger.debug(f"{model_name}, fold {num_fold}")
            print(f"{model_name}, fold {num_fold}")
            num_fold += 1
            # Load the model
            model = utils.load.load_model(model_name, model_path)
            # Handle the custom model
            if isinstance(model, tuple(CustomModel.__subclasses__())):
                # Remove the test data to avoid any saving
                if model.data_test is not None:
                    model.data_test = None
                if model.labels_test is not None:
                    model.labels_test = None

            model.fit(data[train_idx], true_labels[train_idx])

            # Calculate the score
            # Need to take the absolute value because of the make_scorer sklearn convention
            score = np.abs(scorer_func(model, data[test_idx], true_labels[test_idx]))
            num_testsamples = len(true_labels[test_idx])
            # Add the scores
            scores.append(score)
            num_testsamples_list.append(num_testsamples)

        # Maintain the total list
        all_scores.append(scores)
        # Save CV results
        d = {"Scores CV": scores, "Dim test": num_testsamples_list}
        fname = f"{experiment_folder / 'results' / 'GroupShuffleSplit_CV'}_{model_name}_{num_fold}"
        fname += "_holdout" if holdout else ""
        df = pd.DataFrame(d)
        df.to_csv(fname + ".csv")

    pretty_model_names = [pretty_names(name, "model") for name in config_dict["ml"]["model_list"]]

    # Make the boxplot
    sns.boxplot(x=pretty_model_names, y=all_scores, ax=ax, width=0.4)
    # Format the graph
    # ax.set_xticklabels(pretty_names(config_dict['ml']["model_list"], "score"))
    ax.set_xlabel("ML Methods")

    fig = plt.gcf()
    # Save the graph
    if save:
        fname = f"{experiment_folder / 'graphs' / 'boxplot_GroupShuffleSplit_CV'}_{config_dict['ml']['fit_scorer']}"
        fname += "_holdout" if holdout else ""
        save_fig(fig, fname)

    # Close the figure to ensure we start anew
    plt.clf()
    plt.close()
    # Clear keras and TF sessions/graphs etc.
    K.clear_session()


def boxplot_scorer_cv(
    experiment_folder,
    model_list: list[str],
    problem_type: str,
    seed_num: int,
    fit_scorer: str,
    scorer_dict,
    data,
    true_labels,
    nsplits=5,
    save=True,
    holdout=False,
):
    """
    Create a graph of boxplots for all models in the folder, using the specified fit_scorer from the config.

    By default this uses a 5-fold stratified cross validation. Also it saves the list of SHAP values for each of the
    exemplars of each fold
    """
    omicLogger.debug("Creating boxplot_scorer_cv...")
    # Create the plot objects
    fig, ax = plt.subplots()
    # Container for the scores
    all_scores = []
    print(f"Size of data for boxplot: {data.shape}")
    # Create the fold object for CV
    if problem_type == CLASSIFICATION:
        fold_obj = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=seed_num)
    elif problem_type == REGRESSION:
        fold_obj = KFold(n_splits=nsplits, shuffle=True, random_state=seed_num)
    # Loop over the models
    for model_name in model_list:
        # Load the model if trained
        try:
            model_path = glob.glob(f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}")[0]
        except IndexError as e:
            print("The trained model " + str("*" + model_name + "*.pkl") + " is not present")
            raise e

        print(f"Plotting boxplot for {model_name} using {fit_scorer}")
        # Select the scorer
        scorer_func = scorer_dict[fit_scorer]
        # Container for scores for this cross-val for this model
        scores = []
        num_testsamples_list = []
        num_fold = 0

        for train_idx, test_idx in fold_obj.split(data, true_labels):
            omicLogger.debug(f"{model_name}, fold {num_fold}")
            print(f"{model_name}, fold {num_fold}")

            num_fold += 1
            # Load the model
            model = utils.load.load_model(model_name, model_path)
            # Handle the custom model
            if isinstance(model, tuple(CustomModel.__subclasses__())):
                # Remove the test data to avoid any saving
                if model.data_test is not None:
                    model.data_test = None
                if model.labels_test is not None:
                    model.labels_test = None

            model.fit(data[train_idx], true_labels[train_idx])
            # Calculate the score
            # Need to take the absolute value because of the make_scorer sklearn convention
            score = np.abs(scorer_func(model, data[test_idx], true_labels[test_idx]))
            num_testsamples = len(true_labels[test_idx])
            # Add the scores
            scores.append(score)
            num_testsamples_list.append(num_testsamples)

        # Maintain the total list
        all_scores.append(scores)
        # Save CV results
        d = {"Scores CV": scores, "Dim test": num_testsamples_list}
        fname = f"{experiment_folder / 'results' / 'scores_CV'}_{model_name}_{num_fold}"
        fname += "_holdout" if holdout else ""
        df = pd.DataFrame(d)
        df.to_csv(fname + ".csv")

    pretty_model_names = [pretty_names(name, "model") for name in model_list]

    # Make a dataframe
    # df_cv_scores = pd.DataFrame(all_scores, columns=pretty_model_names)
    # fname_all_cv = f"{experiment_folder / 'results' / 'scores_CV_allmodels'}"
    # df_cv_scores.to_csv(fname_all_cv+".csv")
    # print(df_cv_scores)

    # Make the boxplot
    sns.boxplot(x=pretty_model_names, y=all_scores, ax=ax, width=0.4)
    # Format the graph
    # ax.set_xticklabels(pretty_names(config_dict['ml']["model_list"], "score"))
    ax.set_xlabel("ML Methods")

    fig = plt.gcf()
    # Save the graph
    if save:
        fname = f"{experiment_folder / 'graphs' / 'boxplot'}_{fit_scorer}"
        fname += "_holdout" if holdout else ""
        save_fig(fig, fname)

    # Close the figure to ensure we start anew
    plt.clf()
    plt.close()
    # Clear keras and TF sessions/graphs etc.
    K.clear_session()


##########
