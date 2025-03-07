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

from sklearn.preprocessing import normalize
from utils.load import get_data_R2G, load_data, load_model
from utils.ml.preprocessing import apply_ml_preprocessing
from utils.utils import assert_best_model_exists, initial_setup, prof_to_csv
import cProfile
import logging
import numpy as np
import os
import pandas as pd


if __name__ == "__main__":
    """
    Running this script by itself enables for the plots to be made separately from the creation of the models

    Uses the config in the same way as when giving it to run_models.py.
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
        omicLogger.info("Checking for Trained models")
        model_path = assert_best_model_exists(experiment_folder)

        # if the data is R2G then warn the user that the prediction data must have been pre-processed the same way
        if config_dict["data"]["data_type"] == "R2G":
            omicLogger.warning(
                "Previous model was trained with ready to go data. Please ensure that the data being given to this mode has been pre-processed in exactly the same way."
            )

            *_, x_to_predict, _, feature_names = get_data_R2G(
                config_dict, prediction=True
            )
            x_indexes = x_to_predict.index
        else:
            omicLogger.info("Loading Data...")
            x_to_predict, _, features_names = load_data(config_dict, mode="prediction")
            x_indexes = x_to_predict.index

            omicLogger.info("Applying learned ml processing...")
            x_to_predict = apply_ml_preprocessing(
                config_dict, experiment_folder, x_to_predict
            )

        model_name = os.path.basename(model_path).split("_")[0]
        omicLogger.debug("Loading model...")
        model = load_model(model_name, model_path)

        omicLogger.info("Predicting on data...")
        predictions = model.predict(x_to_predict)
        col_names = ["Prediction"]
        if config_dict["ml"]["problem_type"] == "classification":
            predict_proba = normalize(
                model.predict_proba(x_to_predict), axis=1, norm="l1"
            )
            col_names += [f"class_{i}" for i in range(0, predict_proba.shape[1])]
            predictions = np.concatenate(
                (predictions.reshape(-1, 1), predict_proba), axis=1
            )

        omicLogger.info("Saving predictions...")
        predictions = pd.DataFrame(predictions, columns=col_names)
        predictions.index = x_indexes
        predictions.index.name = "SampleID"
        predictions.to_csv(
            experiment_folder / f"{config_dict['prediction']['outfile_name']}.csv",
            index=True,
        )

        omicLogger.info("Process completed.")

    except Exception as e:
        omicLogger.error(e, exc_info=True)
        logging.error(e, exc_info=True)
        raise e

    # save time profile information
    pr.disable()
    prof_to_csv(pr, config_dict)
