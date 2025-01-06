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

import numpy as np
import pandas as pd
import models.model_defs
import utils.ml.class_balancing
import utils.ml.feature_selection
import utils.load
import utils.ml.standardisation
from utils.save import save_transformed_data
import utils.utils
from utils.vars import CLASSIFICATION
import models.models
import mode_plotting
import utils.ml.data_split as ds
import logging
import joblib
import cProfile


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
    ) = utils.utils.initial_setup()

    try:
        omicLogger.info("Loading data...")

        # read the data
        x, y, features_names = utils.load.load_data(config_dict, mode="main")
        omicLogger.info("Data Loaded. Splitting data...")

        if len(x.index.unique()) != x.shape[0]:
            raise ValueError("The sample index/names contain duplicate entries")

        # Split the data in train and test
        x_train, x_test, y_train, y_test = ds.split_data(x, y, config_dict)
        omicLogger.info("Data splitted. Standardising...")

        x_ind_train = x_train.index
        x_ind_test = x_test.index

        # standardise data
        if config_dict["ml"]["standardize"]:
            x_train, SS = utils.ml.standardisation.standardize_data(
                x_train
            )  # fit the standardiser to the training data
            x_test = utils.utils.transform_data(
                x_test, SS
            )  # transform the test data according to the fitted standardiser

            # save the standardiser transformer
            save_name = experiment_folder / "transformer_std.pkl"
            with open(save_name, "wb") as f:
                joblib.dump(SS, f)

        omicLogger.info("Data standardised, transformer saved. Selecting features...")

        # implement feature selection if desired
        if config_dict["ml"]["feature_selection"] is not None:
            x_train, features_names, FS = utils.ml.feature_selection.feat_selection(
                experiment_folder,
                x_train,
                y_train,
                features_names,
                config_dict["ml"]["problem_type"],
                config_dict["ml"]["feature_selection"],
            )
            x_test = FS.transform(x_test)

            # Save the feature selection tranformer
            save_name = experiment_folder / "transformer_fs.pkl"
            with open(save_name, "wb") as f:
                joblib.dump(FS, f)

            omicLogger.info(
                "Features selected, transformer saved. Re-combining data..."
            )
        else:
            print("Skipping Feature selection.")
            omicLogger.info("Skipping feature selection. Re-combining data...")

        # perform class balancing if it is desired
        if config_dict["ml"]["problem_type"] == CLASSIFICATION:
            if config_dict["ml"]["balancing"] == "OVER":
                (
                    x_train,
                    y_train,
                    re_sampled_idxs,
                ) = utils.ml.class_balancing.oversample_data(
                    x_train, y_train, config_dict["ml"]["seed_num"]
                )
                x_ind_train = x_ind_train[re_sampled_idxs]
            elif config_dict["ml"]["balancing"] == "UNDER":
                (
                    x_train,
                    y_train,
                    re_sampled_idxs,
                ) = utils.ml.class_balancing.undersample_data(
                    x_train, y_train, config_dict["ml"]["seed_num"]
                )
                x_ind_train = x_ind_train[re_sampled_idxs]

        # concatenate both test and train into test
        x = np.concatenate((x_train, x_test))
        # y needs to be re-concatenated as the ordering of x may have been changed in splitting
        y = np.concatenate((y_train, y_test))

        # save the transformed input data
        save_transformed_data(
            experiment_folder,
            x,
            y,
            features_names,
            x_test,
            y_test,
            x_ind_train,
            x_ind_test,
        )

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
        models.models.run_models(
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
            mode_plotting.plot_graphs(
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
        best_models = models.models.select_best_model(
            experiment_folder,
            config_dict["ml"]["problem_type"],
            config_dict["ml"]["fit_scorer"],
            collapse_tax,
        )
        utils.utils.copy_best_content(experiment_folder, best_models, collapse_tax)

        omicLogger.info("Process completed.")
    except Exception as e:
        omicLogger.error(e, exc_info=True)
        logging.error(e, exc_info=True)
        raise e

    # save time profile information
    pr.disable()
    utils.utils.prof_to_csv(pr, config_dict)


if __name__ == "__main__":
    # This handles pickling issues when cloning for cross-validation
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

    # Run the models
    main()
