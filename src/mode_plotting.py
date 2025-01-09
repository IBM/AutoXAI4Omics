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

import matplotlib.pyplot as plt
import metrics.metrics as metrics
from plotting.plot_utils import define_plots
from src.utils.load import load_previous_AO_data
import plotting.plots_both
import utils.utils as utils
import cProfile
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
    holdout=False,
):
    """
    Plot graphs as specified by the config. Each plot function is handled separately to be explicit (at the cost of
    length and maintenance). Here you can customize whether you want to graph on train or test based on what arguments
    are given for the data and labels.
    """
    omicLogger.debug("Defining scorers...")
    scorer_dict = metrics.define_scorers(
        config_dict["ml"]["problem_type"], config_dict["ml"]["scorer_list"]
    )

    omicLogger.debug("Defining graphs...")
    plot_dict = define_plots(config_dict["ml"]["problem_type"])

    omicLogger.debug("Begin plotting graphs...")

    # Loop over every plot method we're using
    for plot_method in config_dict["plotting"]["plot_method"]:
        plot_func = plot_dict[plot_method]
        print(plot_method)
        if plot_method == "barplot_scorer":
            plot_func(
                experiment_folder,
                config_dict["ml"]["model_list"],
                config_dict["ml"]["fit_scorer"],
                scorer_dict,
                x_test,
                y_test,
                holdout=holdout,
            )
        elif plot_method == "boxplot_scorer":
            plot_func(
                experiment_folder,
                config_dict["ml"]["model_list"],
                config_dict["ml"]["problem_type"],
                config_dict["ml"]["seed_num"],
                config_dict["ml"]["fit_scorer"],
                scorer_dict,
                x,
                y,
                holdout=holdout,
            )
        elif plot_method == "boxplot_scorer_cv_groupby":
            plot_func(
                experiment_folder, config_dict, scorer_dict, x, y, holdout=holdout
            )
        elif plot_method == "conf_matrix":
            plot_func(
                experiment_folder,
                config_dict["ml"]["model_list"],
                x_test,
                y_test,
                normalize=False,
                holdout=holdout,
            )
        elif plot_method == "corr":
            plot_func(
                experiment_folder,
                config_dict["ml"]["model_list"],
                x_test,
                y_test,
                config_dict["data"]["target"],
                holdout=holdout,
            )
        elif plot_method == "hist":
            plot_func(
                experiment_folder,
                config_dict["ml"]["model_list"],
                x_test,
                y_test,
                config_dict["data"]["target"],
                holdout=holdout,
            )
        elif plot_method == "hist_overlapped":
            plot_func(
                experiment_folder,
                config_dict["ml"]["model_list"],
                x_test,
                y_test,
                config_dict["data"]["target"],
                holdout=holdout,
            )
        elif plot_method == "joint":
            plot_func(
                experiment_folder,
                config_dict["ml"]["model_list"],
                x_test,
                y_test,
                config_dict["data"]["target"],
                holdout=holdout,
            )
        elif plot_method == "joint_dens":
            plot_func(
                experiment_folder,
                config_dict["ml"]["model_list"],
                x_test,
                y_test,
                config_dict["data"]["target"],
                kind="kde",
                holdout=holdout,
            )
        elif plot_method == "permut_imp_test":
            plot_func(
                experiment_folder,
                config_dict["ml"]["seed_num"],
                config_dict["ml"]["model_list"],
                config_dict["ml"]["fit_scorer"],
                config_dict["ml"]["problem_type"],
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
                config_dict["ml"]["problem_type"],
                config_dict["ml"]["model_list"],
                config_dict["plotting"]["explanations_data"],
                feature_names,
                x,
                x_test,
                y_test,
                x_train,
                config_dict["plotting"]["top_feats_shap"],
                holdout=holdout,
            )
        elif plot_method == "roc_curve":
            plot_func(
                experiment_folder,
                config_dict["ml"]["model_list"],
                x_test,
                y_test,
                holdout=holdout,
            )

        # elif plot_method == "shap_force_plots":
        #     plot_func(experiment_folder, config_dict["ml"]["model_list"],config_dict["ml"]["problem_type"], x_test,
        #               y_test, feature_names, x, y, x_train,
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
    plotting.plots_both.K.clear_session()


if __name__ == "__main__":
    """
    Running this script by itself enables for the plots to be made separately from the creation of the models

    Uses the config in the same way as when giving it to run_models.py.
    """

    # init the profiler to time function executions
    pr = cProfile.Profile()
    pr.enable()

    # Do the initial setup
    config_path, config_dict, experiment_folder, omicLogger = utils.initial_setup()

    try:
        omicLogger.info("Loading data...")

        # read in the data
        features_names, x, y, x_train, y_train, x_test, y_test = load_previous_AO_data(
            experiment_folder
        )

        omicLogger.info("Test/train Data Loaded. Begin creating plots...")
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
        )
        omicLogger.info("Process completed.")
    except Exception as e:
        omicLogger.error(e, exc_info=True)
        logging.error(e, exc_info=True)
        raise e

    # save time profile information
    pr.disable()
    utils.prof_to_csv(pr, config_dict)
