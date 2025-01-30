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

from mode_plotting import plot_graphs
from models.models import run_models, select_best_model
from utils.load import get_data_R2G, load_data
from utils.ml.data_split import split_data
from utils.ml.preprocessing import learn_ml_preprocessing
from utils.utils import initial_setup, copy_best_content, prof_to_csv
import cProfile
import logging
import numpy as np
import pandas as pd


def main():
    """
    Central function to tie together preprocessing, running the models, and plotting
    """

    # init the profiler to time function executions
    pr = cProfile.Profile()
    pr.enable()

    # Do the initial setup
    (
        config_path,
        config_dict,
        experiment_folder,
        omicLogger,
    ) = initial_setup()

    try:
        omicLogger.info("Loading data...")

        # Check for R2G
        if config_dict["data"]["data_type"] != "R2G":

            # read the data
            x, y, features_names = load_data(config_dict, mode="main")
            omicLogger.info("Data Loaded. Splitting data...")

            if len(x.index.unique()) != x.shape[0]:
                raise ValueError("The sample index/names contain duplicate entries")

            # Split the data in train and test
            x_train, x_test, y_train, y_test = split_data(x, y, config_dict)
            omicLogger.info("Data splitted. preprocessing data...")

            # Run ml preprocessing
            x, y, features_names, x_train, x_test, y_train = learn_ml_preprocessing(
                config_dict,
                experiment_folder,
                features_names,
                x_train,
                x_test,
                y_train,
                y_test,
            )
        else:

            x, y, x_train, y_train, x_test, y_test, features_names = get_data_R2G(
                config_dict, experiment_folder=experiment_folder
            )
            x_train = x_train.values
            y_train = y_train.values
            x_test = x_test.values
            y_test = y_test.values

        omicLogger.info("Data combined and saved to files. Defining models...")

        print("----------------------------------------------------------")
        print(f"X data shape: {x.shape}")
        print(f"y data shape: {y.shape}")
        print("Dim train:")
        print(x_train.shape)
        print("Dim test:")
        print(x_test.shape)
        print(f"Number of unique values of target y: {len(np.unique(y))}")
        print("----------------------------------------------------------")

        # Load the models we have pre-defined

        omicLogger.info("Models defined. Creating results df holder...")

        # Create dataframes for results
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        omicLogger.info("Holders created. Begin running models...")

        # Run the models
        print("Beginning to run the models")
        run_models(
            config_dict=config_dict,
            model_list=config_dict["ml"]["model_list"],
            df_train=df_train,
            df_test=df_test,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            experiment_folder=experiment_folder,
            fit_scorer=config_dict["ml"]["fit_scorer"],
            hyper_tuning=config_dict["ml"]["hyper_tuning"],
            hyper_budget=config_dict["ml"]["hyper_budget"],
            problem_type=config_dict["ml"]["problem_type"],
            seed_num=config_dict["ml"]["seed_num"],
        )
        print("Finished running models!")

        omicLogger.info("Models trained. Beggining plotting process...")
        # Plot some graphs
        if config_dict["plotting"]["plot_method"] is not None:
            omicLogger.info("Plots defined. Begin plotting graphs...")
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
            )
        else:
            omicLogger.info("No plots desired.")

        if config_dict.get("microbiome") is None:
            collapse_tax = None
        else:
            collapse_tax = config_dict.get("microbiome").get("collapse_tax")

        # Select Best Model
        best_models = select_best_model(
            experiment_folder,
            config_dict["ml"]["problem_type"],
            config_dict["ml"]["fit_scorer"],
            collapse_tax,
        )
        copy_best_content(experiment_folder, best_models, collapse_tax)

        omicLogger.info("Process completed.")
    except Exception as e:
        omicLogger.error(e, exc_info=True)
        logging.error(e, exc_info=True)
        raise e

    # save time profile information
    pr.disable()
    prof_to_csv(pr, config_dict)


if __name__ == "__main__":
    # This handles pickling issues when cloning for cross-validation
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

    # Run the models
    main()
