# Copyright (c) 2025 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from numpy import ndarray
from pandas import DataFrame
from pathlib import Path
from typing import Union
from utils.ml.class_balancing import oversample_data, undersample_data
from utils.ml.feature_selection import feat_selection
from utils.ml.standardisation import standardize_data
from utils.save import save_transformed_data
from utils.utils import assert_data_transformers_exists, transform_data
from utils.vars import CLASSIFICATION
import joblib
import logging
import numpy as np

omicLogger = logging.getLogger("OmicLogger")


def learn_ml_preprocessing(
    config_dict: dict,
    experiment_folder: Path,
    features_names: list[str],
    x_train: Union[DataFrame, ndarray],
    x_test: Union[DataFrame, ndarray],
    y_train: Union[DataFrame, ndarray],
    y_test: Union[DataFrame, ndarray],
):
    x_ind_train = x_train.index
    x_ind_test = x_test.index

    # standardise data
    if config_dict["ml"]["standardize"]:
        omicLogger.info("Standardising data...")
        x_train, SS = standardize_data(
            x_train
        )  # fit the standardiser to the training data
        x_test = transform_data(
            x_test, SS
        )  # transform the test data according to the fitted standardiser

        # save the standardiser transformer
        save_name = experiment_folder / "transformer_std.pkl"
        with open(save_name, "wb") as f:
            joblib.dump(SS, f)
        omicLogger.info("Data standardised, transformer saved.")
    else:
        omicLogger.info("Skipping standardising...")

    # implement feature selection if desired
    if config_dict["ml"]["feature_selection"] is not None:
        omicLogger.info("Selecting features...")
        x_train, features_names, FS = feat_selection(
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

        omicLogger.info("Features selected, transformer saved.")
    else:
        print("Skipping Feature selection.")
        omicLogger.info("Skipping feature selection.")

    # perform class balancing if it is desired
    if config_dict["ml"]["problem_type"] == CLASSIFICATION:
        if config_dict["ml"]["balancing"] == "OVER":
            omicLogger.info("Performing class balancing (OVER sampling)...")
            (
                x_train,
                y_train,
                re_sampled_idxs,
            ) = oversample_data(x_train, y_train, config_dict["ml"]["seed_num"])
            x_ind_train = x_ind_train[re_sampled_idxs]
        elif config_dict["ml"]["balancing"] == "UNDER":
            omicLogger.info("Performing class balancing (UNDER sampling)...")
            (
                x_train,
                y_train,
                re_sampled_idxs,
            ) = undersample_data(x_train, y_train, config_dict["ml"]["seed_num"])
            x_ind_train = x_ind_train[re_sampled_idxs]
        else:
            omicLogger.info("Skipping class balancing...")

    omicLogger.info("Re-combining data...")
    # concatenate both test and train into test
    x = np.concatenate((x_train, x_test))
    # y needs to be re-concatenated as the ordering of x may have been changed in splitting
    y = np.concatenate((y_train, y_test))

    # save the transformed input data
    omicLogger.info("Saving transformed data...")
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

    return x, y, features_names, x_train, x_test, y_train


def apply_ml_preprocessing(
    config_dict: dict, experiment_folder: Path, x_to_transform: DataFrame
) -> DataFrame:
    """Apply learned ml preprocessing

    Parameters
    ----------
    config_dict : dict
        Config dict originally used for training the models
    experiment_folder : Path
        The folder within which the trainign results are in
    x_to_transform : DataFrame
        The dataframe to transform

    Returns
    -------
    DataFrame
        The resulting transformed dataframe
    """

    omicLogger.info("Loading data transformers...")
    # Assert if files exist and load
    SS, FS = assert_data_transformers_exists(experiment_folder, config_dict)

    # apply standardising if not None
    if SS is not None:
        omicLogger.info("Applying trained standardising...")
        x_to_transform = transform_data(x_to_transform, SS)

    # Apply Feature selection if not None
    if FS is not None:
        omicLogger.info("Applying trained feature selector...")
        x_to_transform = FS.transform(x_to_transform)
    return x_to_transform
