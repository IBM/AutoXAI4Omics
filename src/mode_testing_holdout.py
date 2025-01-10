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


from metrics.metrics import evaluate_model, define_scorers
from mode_plotting import plot_graphs
from pathlib import Path
from utils.load import get_data_R2G, load_previous_AO_data, load_data, load_model
from utils.ml.preprocessing import apply_ml_preprocessing
from utils.save import save_results
from utils.utils import (
    assert_best_model_exists,
    get_model_path,
    initial_setup,
    prof_to_csv,
)
import cProfile
import logging
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
        omicLogger.info("Loading data...")
        model_path = assert_best_model_exists(experiment_folder)

        omicLogger.info("Loading previous model input data")
        (features_names, x, y, x_train, y_train, *_) = load_previous_AO_data(
            experiment_folder
        )

        if config_dict["data"]["data_type"] != "R2G":
            omicLogger.info("Loading holdout Data...")
            x_heldout, y_heldout, _ = load_data(config_dict, mode="holdout")

            omicLogger.info("Applying learned ml processing...")
            x_heldout = apply_ml_preprocessing(
                config_dict, experiment_folder, x_heldout
            )
        else:
            # if the data is R2G then warn the user that the holdout data must be pre-processed exactly the same
            omicLogger.warning(
                "Previous model was trained with ready to go data. Please ensure that the data being given to this mode has been pre-processed in exactly the same way."
            )

            *_, x_heldout, y_heldout, features_names = get_data_R2G(
                config_dict, holdout=True
            )

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
            model = load_model(model_name, model_path)

            omicLogger.debug("Evaluating...")

            # Evaluate the best model using all the scores and CV
            performance_results_dict, predictions = evaluate_model(
                model,
                config_dict["ml"]["problem_type"],
                x_train,
                y_train,
                x_heldout,
                y_heldout,
                score_dict=define_scorers(
                    config_dict["ml"]["problem_type"], config_dict["ml"]["scorer_list"]
                ),
            )
            predictions.to_csv(
                results_folder / f"{model_name}_holdout_predictions.csv", index=False
            )

            omicLogger.debug("Saving...")
            # Save the results
            df_performance_results, fname_perfResults = save_results(
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
        plot_graphs(
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
    prof_to_csv(pr, config_dict)
