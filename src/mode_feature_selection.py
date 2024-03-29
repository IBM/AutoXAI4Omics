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
import utils.ml.feature_selection
import utils.load
import utils.ml.standardisation
import utils.utils
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
        x, y, features_names = utils.load.load_data(config_dict)
        omicLogger.info("Data Loaded. Splitting data...")

        if len(x.index.unique()) != x.shape[0]:
            raise ValueError("The sample index/names contain duplicate entries")

        # Split the data in train and test
        x_train, x_test, y_train, y_test = ds.split_data(x, y, config_dict)
        omicLogger.info("Data splitted. Standardising...")

        x_ind_train = x_train.index
        x_ind_test = x_test.index

        # standardise data
        x_train, SS = utils.ml.standardisation.standardize_data(x_train)  # fit the standardiser to the training data
        x_test = utils.utils.transform_data(x_test, SS)  # transform the test data according to the fitted standardiser

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

            omicLogger.info("Features selected, transformer saved. Re-combining data...")
        else:
            print("Skipping Feature selection.")
            omicLogger.info("Skipping feature selection. Re-combining data...")

        # concatenate both test and train into test
        x = np.concatenate((x_train, x_test))
        # y needs to be re-concatenated as the ordering of x may have been changed in splitting
        y = np.concatenate((y_train, y_test))

        # save the transformed input data
        x_df = pd.DataFrame(x, columns=features_names)
        x_df["set"] = "Train"
        x_df["set"].iloc[-x_test.shape[0] :] = "Test"
        x_df.index = list(x_ind_train) + list(x_ind_test)
        x_df.index.name = "SampleID"
        x_df.to_csv(experiment_folder / "transformed_model_input_data.csv", index=True)

        y_df = pd.DataFrame(y, columns=["target"])
        y_df["set"] = "Train"
        y_df["set"].iloc[-y_test.shape[0] :] = "Test"
        y_df.index = list(x_ind_train) + list(x_ind_test)
        y_df.index.name = "SampleID"
        y_df.to_csv(experiment_folder / "transformed_model_target_data.csv", index=True)

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
