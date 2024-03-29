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


from pathlib import Path
from utils.utils import get_model_path
import pandas as pd
import metrics.metrics
import utils.load
import utils.save
import utils.utils
import joblib
import cProfile
import mode_plotting
import logging

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
    ) = utils.utils.initial_setup()

    try:
        omicLogger.info("Loading data...")

        x_heldout, y_heldout, features_names = utils.load.load_data(
            config_dict, load_holdout=None
        )
        omicLogger.info("Heldout Data Loaded. Loading test/train data...")

        x_df = pd.read_csv(
            experiment_folder / "transformed_model_input_data.csv", index_col=0
        )
        x_train = x_df[x_df["set"] == "Train"].iloc[:, :-1].values
        x_test = x_df[x_df["set"] == "Test"].iloc[:, :-1].values
        x = x_df.iloc[:, :-1].values
        features_names = x_df.columns[:-1]

        y_df = pd.read_csv(
            experiment_folder / "transformed_model_target_data.csv", index_col=0
        )
        y_train = y_df[y_df["set"] == "Train"].iloc[:, :-1].values.ravel()
        y_test = y_df[y_df["set"] == "Test"].iloc[:, :-1].values.ravel()
        y = y_df.iloc[:, :-1].values.ravel()
        omicLogger.info("Test/train Data Loaded. Transforming holdout data...")

        with open(experiment_folder / "transformer_std.pkl", "rb") as f:
            SS = joblib.load(f)
        x_heldout = utils.utils.transform_data(
            x_heldout, SS
        )  # transform the holdout data according to the fitted standardiser

        if config_dict["ml"]["feature_selection"] is not None:
            with open(experiment_folder / "transformer_fs.pkl", "rb") as f:
                FS = joblib.load(f)
            x_heldout = FS.transform(x_heldout)

        omicLogger.info("Heldout data transformed. Creating results DataFrame...")
        # Create dataframe for performance results
        df_performance_results = pd.DataFrame()

        # Construct the filepath to save the results
        results_folder = experiment_folder / "results"

        if config_dict["data"]["data_type"] == "microbiome":
            # This is specific to microbiome
            fname = f"scores_{config_dict['microbiome']['collapse_tax']}"
            # Remove or merge samples based on target values (for example merging to categories, if classification)
            if config_dict["microbiome"]["remove_classes"] is not None:
                fname += "_remove"
            elif config_dict["microbiome"]["merge_classes"] is not None:
                fname += "_merge"
        else:
            fname = "scores_"

        # For each model, load it and then compute performance result
        # Loop over the models
        omicLogger.debug("Begin evaluating models...")
        for model_name in config_dict["ml"]["model_list"]:
            omicLogger.debug(f"Evaluate model: {model_name}")
            # Load the model
            model_path = get_model_path(experiment_folder, model_name)

            print(
                f"Plotting barplot for {model_name} using {config_dict['ml']['fit_scorer']}"
            )
            omicLogger.debug("Loading...")
            model = utils.load.load_model(model_name, model_path)

            omicLogger.debug("Evaluating...")

            # Evaluate the best model using all the scores and CV
            performance_results_dict, predictions = metrics.metrics.evaluate_model(
                model,
                config_dict["ml"]["problem_type"],
                x_train,
                y_train,
                x_heldout,
                y_heldout,
                score_dict=metrics.metrics.define_scorers(
                    config_dict["ml"]["problem_type"], config_dict["ml"]["scorer_list"]
                ),
            )
            predictions.to_csv(
                results_folder / f"{model_name}_holdout_predictions.csv", index=False
            )

            omicLogger.debug("Saving...")
            # Save the results
            df_performance_results, fname_perfResults = utils.save.save_results(
                results_folder,
                df_performance_results,
                performance_results_dict,
                model_name,
                fname,
                suffix="_performance_results_holdout",
                save_pkl=False,
                save_csv=True,
            )

            print(
                f"{model_name} evaluation on hold out complete! Results saved at {Path(fname_perfResults).parents[0]}"
            )

        omicLogger.debug("Begin plotting graphs")
        # Central func to define the args for the plots
        mode_plotting.plot_graphs(
            config_dict,
            experiment_folder,
            features_names,
            x,
            y,
            x_train,
            y_train,
            x_heldout,
            y_heldout,
            holdout=True,
        )
        omicLogger.info("Process completed.")

    except Exception as e:
        omicLogger.error(e, exc_info=True)
        logging.error(e, exc_info=True)
        raise e

    # save time profile information
    pr.disable()
    utils.utils.prof_to_csv(pr, config_dict)
