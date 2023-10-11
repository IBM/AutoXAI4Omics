# --------------------------------------------------------------------------
# Licensed Materials - Property of IBM
#
# (C) Copyright IBM Corp. 2019, 2020
# --------------------------------------------------------------------------

# from pathlib import Path
# import re
# import pdb
# import pickle
import numpy as np

# import scipy.sparse
import pandas as pd
import matplotlib.pyplot as plt

# import matplotlib.colors as colors
# import matplotlib.ticker as ticker
# import matplotlib.image as mp_img
# from sklearn.model_selection import cross_val_score
import models.models as models
from plotting.plots import define_plots
import plotting.plots
import utils.utils as utils
from models.custom_model import CustomModel

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
# )

# import imblearn
import cProfile

##########
# from data_processing import *
import utils.data_processing as data_processing
import logging


omicLogger = logging.getLogger("OmicLogger")


def plot_graphs(
    config_dict,
    experiment_folder,
    feature_names,
    x,
    y,
    x_train,
    y_train,
    x_test,
    y_test,
    scorer_dict,
    holdout=False,
):
    """
    Plot graphs as specified by the config. Each plot function is handled separately to be explicit (at the cost of
    length and maintenance). Here you can customize whether you want to graph on train or test based on what arguments
    are given for the data and labels.
    """
    omicLogger.debug("Defining graphs...")
    plot_dict = define_plots(config_dict["ml"]["problem_type"])

    omicLogger.debug("Begin plotting graphs...")

    # Loop over every plot method we're using
    for plot_method in config_dict["plotting"]["plot_method"]:
        plot_func = plot_dict[plot_method]
        print(plot_method)
        # Hand-crafted passing the arguments in because over-engineering
        # Don't judge me (I'm not a big **kwargs fan)
        if plot_method == "barplot_scorer":
            plot_func(
                experiment_folder,
                config_dict,
                scorer_dict,
                x_test,
                y_test,
                holdout=holdout,
            )
        elif plot_method == "boxplot_scorer":
            plot_func(experiment_folder, config_dict, scorer_dict, x, y, holdout=holdout)
        elif plot_method == "boxplot_scorer_cv_groupby":
            plot_func(experiment_folder, config_dict, scorer_dict, x, y, holdout=holdout)
        elif plot_method == "conf_matrix":
            plot_func(
                experiment_folder,
                config_dict,
                x_test,
                y_test,
                normalize=False,
                holdout=holdout,
            )
        elif plot_method == "corr":
            plot_func(
                experiment_folder,
                config_dict,
                x_test,
                y_test,
                config_dict["data"]["target"],
                holdout=holdout,
            )
        elif plot_method == "hist":
            plot_func(
                experiment_folder,
                config_dict,
                x_test,
                y_test,
                config_dict["data"]["target"],
                holdout=holdout,
            )
        elif plot_method == "hist_overlapped":
            plot_func(
                experiment_folder,
                config_dict,
                x_test,
                y_test,
                config_dict["data"]["target"],
                holdout=holdout,
            )
        elif plot_method == "joint":
            plot_func(
                experiment_folder,
                config_dict,
                x_test,
                y_test,
                config_dict["data"]["target"],
                holdout=holdout,
            )
        elif plot_method == "joint_dens":
            plot_func(
                experiment_folder,
                config_dict,
                x_test,
                y_test,
                config_dict["data"]["target"],
                kind="kde",
                holdout=holdout,
            )
        elif plot_method == "permut_imp_test":
            plot_func(
                experiment_folder,
                config_dict,
                scorer_dict,
                feature_names,
                x_test,
                y_test,
                config_dict["plotting"]["top_feats_permImp"],
                cv="prefit",
                holdout=holdout,
            )
        elif plot_method == "shap_plots":
            plot_func(
                experiment_folder,
                config_dict,
                feature_names,
                x,
                x_test,
                y_test,
                x_train,
                config_dict["plotting"]["top_feats_shap"],
                holdout=holdout,
            )
        elif plot_method == "roc_curve":
            plot_func(experiment_folder, config_dict, x_test, y_test, holdout=holdout)

        # elif plot_method == "shap_force_plots":
        #     plot_func(experiment_folder, config_dict, x_test, y_test, feature_names, x, y, x_train,
        #                       data_forexplanations="all", top_exemplars=0.4, save=True)
        # elif plot_method == "permut_imp_alldata":
        #     plot_func(experiment_folder, config_dict, scorer_dict, feature_names, x, y,
        #                       config_dict['plotting']["top_feats_permImp"], cv='prefit')
        # elif plot_method == "permut_imp_train":
        #     plot_func(experiment_folder, config_dict, scorer_dict, feature_names, x_train, y_train,
        #                       config_dict['plotting']["top_feats_permImp"], cv='prefit')
        # elif plot_method == "permut_imp_5cv":
        #     plot_func(experiment_folder, config_dict, scorer_dict, feature_names, x, y,
        #                       config_dict['plotting']["top_feats_permImp"], cv=5)

    omicLogger.debug("Plotting completed")
    # Clear everything
    plt.clf()
    plt.close()
    # Clear keras and TF sessions/graphs etc.
    plotting.plots.tidy_tf()


if __name__ == "__main__":
    """
    Running this script by itself enables for the plots to be made separately from the creation of the models

    Uses the config in the same way as when giving it to run_models.py.
    """
    # Do the initial setup
    config_path, config_dict = utils.initial_setup()

    # init the profiler to time function executions
    pr = cProfile.Profile()
    pr.enable()

    # Set the global seed
    np.random.seed(config_dict["ml"]["seed_num"])

    # Create the folders needed
    experiment_folder = utils.create_experiment_folders(config_dict, config_path)

    # Set up process logger
    omicLogger = utils.setup_logger(experiment_folder)
    try:
        omicLogger.info("Loading data...")

        # read in the data
        x_df = pd.read_csv(experiment_folder / "transformed_model_input_data.csv", index_col=0)
        x_train = x_df[x_df["set"] == "Train"].iloc[:, :-1].values
        x_test = x_df[x_df["set"] == "Test"].iloc[:, :-1].values
        x = x_df.iloc[:, :-1].values
        features_names = x_df.columns[:-1]

        y_df = pd.read_csv(experiment_folder / "transformed_model_target_data.csv", index_col=0)
        y_train = y_df[y_df["set"] == "Train"].iloc[:, :-1].values.ravel()
        y_test = y_df[y_df["set"] == "Test"].iloc[:, :-1].values.ravel()
        y = y_df.iloc[:, :-1].values.ravel()
        omicLogger.info("Test/train Data Loaded. Defining scorers...")

        # Select only the scorers that we want
        scorer_dict = models.define_scorers(config_dict["ml"]["problem_type"])
        scorer_dict = {k: scorer_dict[k] for k in config_dict["ml"]["scorer_list"]}
        omicLogger.info("All scorers defined. Defining plots...")

        # Pickling doesn't inherit the self.__class__.__dict__, just self.__dict__
        # So set that up here
        # Other option is to modify cls.__getstate__
        for model_name in config_dict["ml"]["model_list"]:
            if model_name in CustomModel.custom_aliases:
                CustomModel.custom_aliases[model_name].setup_cls_vars(config_dict["ml"], experiment_folder)
        omicLogger.info("Plots defined. Begin creating plots...")
        # Central func to define the args for the plots
        plot_graphs(
            config_dict,
            experiment_folder,
            features_names,
            x,
            y,
            x_train,
            y_train,
            x_test,
            y_test,
            scorer_dict,
        )
        omicLogger.info("Process completed.")
    except Exception as e:
        omicLogger.error(e, exc_info=True)
        logging.error(e, exc_info=True)
        raise e

    # save time profile information
    pr.disable()
    csv = data_processing.prof_to_csv(pr)
    with open(
        f"{config_dict['data']['save_path']}results/{config_dict['data']['name']}/time_profile.csv",
        "w+",
    ) as f:
        f.write(csv)
