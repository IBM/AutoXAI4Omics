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

from utils.ml.preprocessing import learn_ml_preprocessing
import cProfile
import logging
import utils.load
import utils.ml.data_split as ds
import utils.utils


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

        # If the data is R2G then it has already had feature selection done and not suitable for this mode
        if config_dict["data"]["data_type"] == "R2G":
            raise ValueError(
                "Configs with data:data_type=R2G can not be used in feature_selection mode"
            )

        # read the data
        x, y, features_names = utils.load.load_data(config_dict, mode="main")
        omicLogger.info("Data Loaded. Splitting data...")

        if len(x.index.unique()) != x.shape[0]:
            raise ValueError("The sample index/names contain duplicate entries")

        # Split the data in train and test
        x_train, x_test, y_train, y_test = ds.split_data(x, y, config_dict)
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
