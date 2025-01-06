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

import json
from pathlib import Path

from numpy import ndarray
import pandas as pd
from models.custom_model import CustomModel


import joblib
from omics import geneExp, metabolomic, microbiome, tabular

import logging

omicLogger = logging.getLogger("OmicLogger")


def load_model(model_name, model_path):
    """
    Load a previously saved and trained model. Uses joblib's version of pickle.
    """
    print("Model path: ")
    print(model_path)
    print("Model ")
    print()

    if model_name in CustomModel.custom_aliases:
        # Remove .pkl here, it will be handled later
        model_path = model_path.replace(".pkl", "")

        try:
            model = CustomModel.custom_aliases[model_name].load_model(model_path)
        except Exception as e:
            print("The trained model " + model_name + " is not present")
            raise e
    else:
        # Load a previously saved model (using joblib's pickle)
        with open(model_path, "rb") as f:
            model = joblib.load(f)
    return model


def load_config(config_path: str) -> dict:
    """
    Load a JSON file (general function, but we use it for configs)
    """
    with open(config_path) as json_file:
        config_dict = json.load(json_file)
    return config_dict


def get_non_omic_data(
    path_file: Path,
    target: str,
    metadata_path: Path | str | None,
    prediction: bool = False,
):
    """
    Read the input files and return X, y (target) and the feature_names
    """
    omicLogger.debug("Inserting data into DataFrames...")
    # Read the data
    data = pd.read_csv(path_file)

    # check if the first column is meant to be the index
    # Assumption is that if the first column name is empty/none then it is meant to be an index
    if ("Unnamed" in data.columns[0]) or (data.columns[0] is None):
        data.set_index(data.columns[0], inplace=True)
        data.index.name = None

    print("Data dimension: " + str(data.shape))

    if not prediction:
        # Check if the target is in a separate file or in the same data
        if not metadata_path or metadata_path == "":
            y = data[target].values
            data_notarget = data.drop(target, axis=1)

        else:  # it assumes the data does not contain the target column
            # Read the metadata file
            metadata = pd.read_csv(metadata_path, index_col=0)
            y = metadata[target].values
            data_notarget = data

        features_names = data_notarget.columns
        x = data_notarget  # .values

        # Check the data and labels are the right size
        assert len(x) == len(y)

    else:
        try:
            data_notarget = data.drop(target, axis=1)
        except Exception:
            data_notarget = data

        x = data_notarget
        features_names = data_notarget.columns
        y = None

    return x, y, features_names


def load_data_prediction(config_dict: dict):
    omicLogger.debug("Loading prediction data")

    if config_dict["data"]["data_type"] == "microbiome":
        x, _, features_names = microbiome.get_data_microbiome_trained(
            config_dict, holdout=False, prediction=True
        )

    elif config_dict["data"]["data_type"] == "gene_expression":
        x, _, features_names = geneExp.get_data_gene_expression_trained(
            config_dict, holdout=False, prediction=True
        )

    elif config_dict["data"]["data_type"] == "metabolomic":
        x, _, features_names = metabolomic.get_data_metabolomic_trained(
            config_dict, holdout=False, prediction=True
        )

    elif config_dict["data"]["data_type"] == "tabular":
        x, _, features_names = tabular.get_data_tabular_trained(
            config_dict, holdout=False, prediction=True
        )

    else:
        x, _, features_names = get_non_omic_data(
            config_dict["prediction"]["file_path"],
            config_dict["data"]["target"],
            "",
            True,
        )

    return x, features_names


def load_data_holdout(config_dict: dict):
    omicLogger.debug("Training loaded. Loading holdout data...")
    if config_dict["data"]["data_type"] == "microbiome":
        # This reads and preprocesses microbiome data using calour library --
        # it would be better to change this preprocessing so that it is not dependent from calour
        x_heldout, y_heldout, features_names = microbiome.get_data_microbiome(
            config_dict["data"]["file_path_holdout_data"],
            config_dict["data"]["metadata_file_holdout_data"],
            config_dict,
        )
    elif config_dict["data"]["data_type"] == "gene_expression":
        # This reads and preprocesses microbiome data using calour library --
        # it would be better to change this preprocessing so that it is not dependent from calour
        x_heldout, y_heldout, features_names = geneExp.get_data_gene_expression(
            config_dict, holdout=True
        )
    elif config_dict["data"]["data_type"] == "metabolomic":
        x_heldout, y_heldout, features_names = metabolomic.get_data_metabolomic(
            config_dict, holdout=True
        )
    elif config_dict["data"]["data_type"] == "tabular":
        x_heldout, y_heldout, features_names = tabular.get_data_tabular(
            config_dict, holdout=True
        )
    else:
        # At the moment for all the other data types, for example metabolomics, we have not implemented preprocessing
        # except for standardisation with StandardScaler()
        x_heldout, y_heldout, features_names = get_non_omic_data(
            config_dict["data"]["file_path_holdout_data"],
            config_dict["data"]["target"],
            config_dict["data"]["metadata_file_holdout_data"],
        )

    return x_heldout, y_heldout, features_names


def load_data_main(config_dict: dict):
    omicLogger.debug("Loading training data...")

    if config_dict["data"]["data_type"] == "microbiome":
        # This reads and preprocesses microbiome data using calour library -- it would be better to change this
        # preprocessing so that it is not dependent from calour
        x, y, features_names = microbiome.get_data_microbiome(
            config_dict["data"]["file_path"],
            config_dict["data"]["metadata_file"],
            config_dict,
        )
    elif config_dict["data"]["data_type"] == "gene_expression":
        # This reads and preprocesses microbiome data using calour library -- it would be better to change this
        # preprocessing so that it is not dependent from calour
        x, y, features_names = geneExp.get_data_gene_expression(config_dict)
    elif config_dict["data"]["data_type"] == "metabolomic":
        x, y, features_names = metabolomic.get_data_metabolomic(config_dict)
    elif config_dict["data"]["data_type"] == "tabular":
        x, y, features_names = tabular.get_data_tabular(config_dict)
    else:
        # At the moment for all the other data types, for example metabolomics, we have not implemented preprocessing
        # except for standardisation with StandardScaler()
        x, y, features_names = get_non_omic_data(
            config_dict["data"]["file_path"],
            config_dict["data"]["target"],
            config_dict["data"]["metadata_file"],
        )

    return x, y, features_names


def load_data(
    config_dict: dict, load_holdout: bool | None = False, load_prediction: bool = False
):
    """
    A function to handel all of the loading of the data presented in the config file

    Parameters
    ----------
    load_holdout : bool or None
        A var used to control which data to be loaded if: none, returns only the holdout data; True, returns both the
        non-holdout and the holdout data; False, returns only the non-holdout data.
    load_prediction : bool
        If true will not load the main or hold out datasets but the data to be predicted

    Returns
    -------
    x : Numpy array
        if load_prediction was true then this is the data to be predicted on other wise it is the data to train/test on
    y : Numpy array
        if load_prediction was true then this is not returned on other wise it is the y values for the train/test data
    features_names : list of str
        The list of feature names.
    x_heldout : Numpy array
        if load_prediction was true and load_holdout was None/True then this is the heldout data
    y_heldout : Numpy array
        if load_prediction was true and load_holdout was None/True then this is the y values for the heldout data
    """
    omicLogger.debug("Data load inititalised")

    if not load_prediction:
        if load_holdout is not None:
            x, y, features_names = load_data_main(config_dict)

        if (load_holdout is None) or load_holdout:
            x_heldout, y_heldout, features_names = load_data_holdout(config_dict)

        omicLogger.debug("Load completed")
        if load_holdout is None:
            return x_heldout, y_heldout, features_names
        elif load_holdout:
            return x, y, x_heldout, y_heldout, features_names
        else:
            return x, y, features_names
    else:
        x, features_names = load_data_prediction(config_dict)
        return x, features_names


def load_transformed_data(
    experiment_folder: Path,
) -> tuple[list[str], ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
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
    return features_names, x, y, x_train, y_train, x_test, y_test
