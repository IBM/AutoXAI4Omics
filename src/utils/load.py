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
from omics import geneExp, metabolomic, microbiome, tabular
from pathlib import Path
from typing import Literal
import joblib
import json
import logging
import pandas as pd
from utils.save import save_transformed_data

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
) -> tuple[pd.DataFrame, ndarray, list[str]]:
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
        features_names = data_notarget.columns.to_list()
        y = None

    return x, y, features_names


def load_data_prediction(config_dict: dict) -> tuple[pd.DataFrame, list[str]]:
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


def load_data_holdout(config_dict: dict) -> tuple[pd.DataFrame, ndarray, list[str]]:
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


def load_data_main(config_dict: dict) -> tuple[pd.DataFrame, ndarray, list[str]]:
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


def get_data_R2G(
    config_dict: dict,
    prediction: bool = False,
    holdout: bool = False,
    experiment_folder: Path | None = None,
) -> tuple[
    pd.DataFrame | None,
    pd.Series | None,
    pd.DataFrame | None,
    pd.Series | None,
    pd.DataFrame,
    pd.Series | None,
    list[str],
]:
    """A function to load R2G data files

    Parameters
    ----------
    config_dict : dict
        The config dict containing the relevant data information
    prediction : bool, optional
        A boll to indicate if teh dataset being loaded is for prediction or not, by default False
    holdout : bool, optional
        A bool to indicate if the dataset being loaded is for holdout testing or not, by default False
    experiment_folder : Path | none, optional
        A Path in which to save the AO inputs to if provided, byt default none


    Returns
    -------
    tuple[ pd.DataFrame | None, pd.Series | None, pd.DataFrame | None, pd.Series | None, pd.DataFrame, pd.Series | None, list[str] ]
        Return the X,y, x_train, y_train, x_test,y_test and feature names. If prediction is True all bar x_test and feature_names will be None. The same if holdout is true with teh exception of y_test

    Raises
    ------
    ValueError
        Is raised if the data_path found is None
    """
    # TODO: 5. input validation
    # TODO: 7. write test

    # extract data path to load
    if prediction:
        data_path = config_dict["prediction"]["file_path"]
    elif holdout:
        data_path = config_dict["data"]["file_path_holdout_data"]
    else:
        data_path = config_dict["data"]["file_path"]

    if data_path is None:
        raise ValueError("Recieved None for data_path when loading R2G dataset")

    # load df
    r2g_df = pd.read_csv(data_path, index_col=0)

    # validate dataframe
    validate_r2g_dataset(r2g_df)

    # extract out test set as should be present for all modes
    X_test_df = r2g_df[r2g_df["set"] == "test"].drop(columns="set")

    # check if in prediction loading mode
    if not prediction:
        # if not then extract test labels
        y_test = X_test_df["label"]
    else:
        # else set to None
        y_test = None

    # drop the label col
    X_test_df.drop(columns="label", inplace=True)

    # extract the feature names
    feature_names = X_test_df.columns.to_list()

    # Check if in main loading mode
    if not (prediction or holdout):
        # then extract the "train" subset
        X_train_df = r2g_df[r2g_df["set"] == "train"].drop(columns="set")
        y_train = X_train_df["label"]
        X_train_df.drop(columns="label", inplace=True)
        X = r2g_df.drop(columns=["set", "label"])
        y = r2g_df["label"]

        # save AO input files
        if experiment_folder:
            save_transformed_data(
                experiment_folder,
                X,
                y,
                feature_names,
                X_test_df,
                y_test,
                X_train_df.index,
                X_test_df.index,
            )
    else:
        # else set to None
        X_train_df = y_train = X = y = None

    return X, y, X_train_df, y_train, X_test_df, y_test, feature_names


def validate_r2g_dataset(r2g_df: pd.DataFrame):
    """Assert that the provided dataframe is a valid R2G dataset

    Parameters
    ----------
    r2g_df : pd.DataFrame
        The dataframe to be validated

    Raises
    ------
    ValueError
        Is raised if the dataframe does not have a 'set' or 'label' column
    ValueError
        Is raised if the dataframe has values other than 'test' and 'train' in its 'label' column
    """
    if not ("label" in r2g_df.columns and "set" in r2g_df.columns):
        raise ValueError(
            "A R2G dataset must have both a 'set' column and a 'label' column. At least one was not found."
        )

    if r2g_df["set"].unique().tolist() != ["test", "train"]:
        raise ValueError(
            "The 'set' column of a R2G dataset must only contain 'train' or 'test'"
        )


def load_data(
    config_dict: dict, mode: Literal["main", "holdout", "prediction"] = "main"
) -> tuple[pd.DataFrame, ndarray, list[str]]:
    """A function to handel all of the loading of the data presented in the config file.

    Parameters
    ----------
    config_dict : dict
        The dict containign the information needed to load the data
    mode : Literal[&quot;main&quot;, &quot;holdout&quot;, &quot;prediction&quot;], optional
        The context of the data loading to be done in, by default "main"

    Returns
    -------
    tuple[pd.DataFrame,ndarray,list[str]]
        The dataset along with the lables and a list of the feature names

    Raises
    ------
    ValueError
        is raised if the value for mode is not a valid entry
    """
    omicLogger.debug("Data load inititalised")

    if mode == "prediction":
        x, features_names = load_data_prediction(config_dict)
        return x, None, features_names
    elif mode == "holdout":
        x_heldout, y_heldout, features_names = load_data_holdout(config_dict)
        return x_heldout, y_heldout, features_names
    elif mode == "main":
        x, y, features_names = load_data_main(config_dict)
        return x, y, features_names
    else:
        raise ValueError(
            f"Unrecognised value for mode, valid values must be one of 'main', 'holdout','predication'. recieved: {mode=}"
        )


def load_previous_AO_data(
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
