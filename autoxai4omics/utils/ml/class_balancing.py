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

from numpy import ndarray
from pandas.core.frame import DataFrame
from typing import Union
import imblearn
import logging

omicLogger = logging.getLogger("OmicLogger")


def oversample_data(
    x_train: Union[ndarray, DataFrame],
    y_train: Union[ndarray, DataFrame],
    seed: int = 29292,
) -> tuple[ndarray, ndarray, ndarray]:
    """Given the training set it has a class imbalance problem, this will over sample the training data to balance out
    the classes

    Parameters
    ----------
    x_train : Union[ndarray, DataFrame]
        The training data that needs to be re-sampled
    y_train : Union[ndarray, DataFrame]
        The train labels to be re-sampled
    seed : int, optional
        The seed to control the random sampling, by default 29292

    Returns
    -------
    tuple[ndarray,ndarray,ndarray]
        A tuple containing the re-sampled training data, labels plus and the indicies of what original samples have been
         used

    Raises
    ------
    TypeError
        is raised if the seed is not an int
    TypeError
        is raised if x_train or y_train is not an ndarray or a pandas DataFrame
    ValueError
        is raised if x_train and y_train dont have the same number of rows
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, recieved {type(seed)}")

    if not isinstance(x_train, (ndarray, DataFrame)):
        raise TypeError(
            f"x_train must be either a ndarray or a DataFrame. Recieved: {type(x_train)}"
        )

    if not isinstance(y_train, (ndarray, DataFrame)):
        raise TypeError(
            f"y_train must be either a ndarray or a DataFrame. Recieved: {type(y_train)}"
        )

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"x_train and y_train have different numbers of rows - "
            f"x_train: ({x_train.shape[0]}) and y_train: ({y_train.shape[0]})"
        )

    omicLogger.debug("Oversampling data...")
    # define oversampling strategy
    oversample = imblearn.over_sampling.RandomOverSampler(
        random_state=seed, sampling_strategy="not majority"
    )
    # fit and apply the transform
    x_resampled, y_resampled = oversample.fit_resample(x_train, y_train)
    omicLogger.info(f"X train data after oversampling shape: {x_resampled.shape}")
    omicLogger.info(f"y train data after oversampling shape: {y_resampled.shape}")

    return x_resampled, y_resampled, oversample.sample_indices_


def undersample_data(
    x_train: Union[ndarray, DataFrame],
    y_train: Union[ndarray, DataFrame],
    seed: int = 29292,
) -> tuple[ndarray, ndarray, ndarray]:
    """Given the training set it has a class imbalance problem, this will under sample the training data to balance out
    theclasses

    Parameters
    ----------
    x_train : Union[ndarray, DataFrame]
        The training data that needs to be re-sampled
    y_train : Union[ndarray, DataFrame]
        The train labels to be re-sampled
    seed : int, optional
        The seed to control the random sampling, by default 29292

    Returns
    -------
    tuple[ndarray,ndarray,ndarray]
        A tuple containing the re-sampled training data, labels plus and the indicies of what original samples have been
         used

    Raises
    ------
    TypeError
        is raised if the seed is not an int
    TypeError
        is raised if x_train or y_train is not an ndarray or a pandas DataFrame
    ValueError
        is raised if x_train and y_train dont have the same number of rows
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, recieved {type(seed)}")

    if not isinstance(x_train, (ndarray, DataFrame)):
        raise TypeError(
            f"x_train must be either a ndarray or a DataFrame. Recieved: {type(x_train)}"
        )

    if not isinstance(y_train, (ndarray, DataFrame)):
        raise TypeError(
            f"x_train must be either a ndarray or a DataFrame. Recieved: {type(x_train)}"
        )

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"x_train and y_train have different numbers of rows - "
            f"x_train: ({x_train.shape[0]}) and y_train: ({y_train.shape[0]})"
        )
    omicLogger.debug("Undersampling data...")
    # define undersampling strategy
    oversample = imblearn.under_sampling.RandomUnderSampler(
        random_state=seed, sampling_strategy="not minority"
    )
    # fit and apply the transform
    x_resampled, y_resampled = oversample.fit_resample(x_train, y_train)
    omicLogger.info(f"X train data after undersampling shape: {x_resampled.shape}")
    omicLogger.info(f"y train data after undersampling shape: {y_resampled.shape}")

    return x_resampled, y_resampled, oversample.sample_indices_
