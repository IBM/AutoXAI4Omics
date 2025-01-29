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

from models.custom_model import CustomModel
from numpy import ndarray
from pathlib import Path
from utils.vars import CLASSIFICATION
import joblib
import json
import logging
import pandas as pd

omicLogger = logging.getLogger("OmicLogger")


def save_config(experiment_folder, config_path, config_dict):
    """
    Save the config into the results folder for easy access (storage is cheap right?)
    """
    # Construct the file name
    fname = experiment_folder / config_path.name
    with open(fname, "w") as outfile:
        json.dump(json.loads(config_dict), outfile, indent=4)


def save_exemplars_SHAP_values(
    config_dict,
    experiment_folder,
    feature_names,
    model_name,
    class_names,
    exemplars_selected,
    fold_id,
):
    # Deal with classification differently, classification has shap values for each class
    # Get the SHAP values (global impact) sorted from the highest to the lower (absolute value)
    if config_dict["ml"]["problem_type"] == CLASSIFICATION:
        # XGBoost for binary classification seems to return the SHAP values only for class 1
        if model_name == "xgboost" and len(class_names) == 2:
            df_exemplars = pd.DataFrame(data=exemplars_selected, columns=feature_names)
            fname_exemplars = f"{experiment_folder / 'results' / 'exemplars_SHAP_values'}_{model_name}_{fold_id}"
            df_exemplars.to_csv(fname_exemplars + ".txt")

        # When class > 2 (or class > 1 for all the models except XGBoost) SHAP return a list of SHAP value matrices.
        # One for each class.
        else:
            print(type(exemplars_selected))
            print(len(exemplars_selected))
            print(len(exemplars_selected) == len(class_names))

            for i in range(len(exemplars_selected)):
                print("Class: " + str(i))
                print("Class name: " + str(class_names[i]))
                df_exemplars = pd.DataFrame(
                    data=exemplars_selected[i], columns=feature_names
                )
                fname_exemplars = (
                    f"{experiment_folder / 'results' / 'exemplars_SHAP_values'}_{model_name}_"
                    + f"{class_names[i]}_{i}_{fold_id}"
                )
                df_exemplars.to_csv(fname_exemplars + ".txt")

    # Deal with regression
    else:
        df_exemplars = pd.DataFrame(data=exemplars_selected, columns=feature_names)
        fname_exemplars = f"{experiment_folder / 'results' / 'exemplars_SHAP_values'}_{model_name}_{fold_id}"
        df_exemplars.to_csv(fname_exemplars + ".txt")


def save_explainer(experiment_folder, model_name, explainer):
    save_name = (
        f"{experiment_folder / 'models' / 'explainers' / 'shap'}_{model_name}.pkl"
    )
    with open(save_name, "wb") as f:
        joblib.dump(explainer, f)


def save_fig(fig, fname, dpi=200, fig_format="png"):
    omicLogger.debug(f"Saving figure ({fname})to file...")
    print(f"Save location: {fname}.{fig_format}")
    fig.savefig(
        f"{fname}.{fig_format}",
        dpi=dpi,
        format=fig_format,
        bbox_inches="tight",
        transparent=False,
    )


def save_results(
    results_folder,
    df,
    score_dict,
    model_name,
    fname,
    suffix=None,
    save_pkl=False,
    save_csv=True,
):
    """
    Store the results of the latest model and save this to csv
    """
    omicLogger.debug("Save results to file...")

    # df = df.append(pd.Series(score_dict, name=model_name))
    # TODO: remove above if below works
    df = pd.concat([df, pd.DataFrame.from_records([score_dict], index=[model_name])])
    fname = str(results_folder / fname)
    # Add a suffix to the filename if provided
    if suffix is not None:
        fname += suffix
    # Save as a csv
    if save_csv:
        df.to_csv(fname + ".csv", index_label="model")
    # Pickle using pandas internal access to it
    if save_pkl:
        df.to_pickle(fname + ".pkl")
    return df, fname


def save_model(experiment_folder, model, model_name):
    """
    Save a given model to the model folder
    """
    omicLogger.debug("Saving model...")
    model_folder = experiment_folder / "models"
    # THe CustomModels handle themselves
    if model_name not in CustomModel.custom_aliases:
        print(f"Saving {model_name} model")
        save_name = model_folder / f"{model_name}_best.pkl"
        with open(save_name, "wb") as f:
            joblib.dump(model, f)
    else:  # hat: added this
        model.save_model()


def save_transformed_data(
    experiment_folder: Path,
    x: ndarray,
    y: ndarray,
    features_names: list[str],
    x_test: ndarray,
    y_test: ndarray,
    x_ind_train,
    x_ind_test,
):
    x_df = pd.DataFrame(x, columns=features_names)
    x_df["set"] = "Train"
    x_df["set"].iloc[-x_test.shape[0] :] = "Test"
    x_df.index = list(x_ind_train) + list(x_ind_test)
    x_df.index.name = "SampleID"
    save_path = experiment_folder / "transformed_model_input_data.csv"
    omicLogger.info(f"saving input data to: {save_path}")
    x_df.to_csv(save_path, index=True)

    y_df = pd.DataFrame(y, columns=["target"])
    y_df["set"] = "Train"
    y_df["set"].iloc[-y_test.shape[0] :] = "Test"
    y_df.index = list(x_ind_train) + list(x_ind_test)
    y_df.index.name = "SampleID"
    save_path = experiment_folder / "transformed_model_target_data.csv"
    omicLogger.info(f"saving target data to: {save_path}")
    y_df.to_csv(save_path, index=True)
