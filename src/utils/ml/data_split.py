import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, train_test_split
import logging
from utils.vars import CLASSIFICATION, REGRESSION
from typing import Union
from numpy import ndarray
from pandas.core.frame import DataFrame
import os

omicLogger = logging.getLogger("OmicLogger")


################### SPLIT DATA ###################


def split_data(x, y, config_dict):
    """
    Split the data according to the config (i.e normal split or stratify by groups)
    """

    omicLogger.debug("Splitting data...")
    # Split the data in train and test
    if config_dict["ml"]["stratify_by_groups"] == "Y":
        x_train, x_test, y_train, y_test = strat_split(
            x,
            y,
            config_dict["data"]["metadata_file"],
            config_dict["ml"]["groups"],
            config_dict["ml"]["test_size"],
            config_dict["ml"]["seed_num"],
        )

    else:
        x_train, x_test, y_train, y_test = std_split(
            x,
            y,
            config_dict["ml"]["problem_type"],
            config_dict["ml"]["test_size"],
            config_dict["ml"]["seed_num"],
        )

    return x_train, x_test, y_train, y_test


def strat_split(
    x: Union[ndarray, DataFrame],
    y: Union[ndarray, DataFrame],
    meta_file: str,
    group_name: str,
    test_size: float = 0.2,
    seed: int = 29292,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """split the data according to stratification

    Parameters
    ----------
    x : Union[ndarray, DataFrame]
        The data to be split
    y : Union[ndarray, DataFrame]
        The corresponding labels of the data to be split
    meta_file : str
        a str file path to the meta data to be used for the stratification
    group_name : str
        the group (column name) to split the data by in the metadata file
    test_size : float
        a float to determine the size of the test set, by default 0.2
    seed : int
        the seed to control the randomisation, by default 29292

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, ndarray]
        returns the x_train, x_test, y_train and y_test to be used

    Raises
    ------
    TypeError
        is raised if x_full or y_full is not a ndarray of Dataframe
    ValueError
        is raised if x_full and y_full dont have the same number of rows
    TypeError
        is rasied if test_size is not a float
    ValueError
        is raised if test_size is not between 0.0 and 1.0
    TypeError
        is raised if seed is not an int
    TypeError
        is raised if meta_file is not a str
    FileNotFoundError
        is raised if the file given in meta_file does not exist
    IsADirectoryError
        is raised if the file given in meta_file is actually a directory
    TypeError
        is raised if group_name is not a str
    ValueError
        is raised if group_name is not present in the columns of the meta_file
    """
    if not isinstance(x, (ndarray, DataFrame)):
        raise TypeError(
            f"x must be either a ndarray or a DataFrame. Recieved: {type(x)}"
        )

    if not isinstance(y, (ndarray, DataFrame)):
        raise TypeError(
            f"x must be either a ndarray or a DataFrame. Recieved: {type(y)}"
        )

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x and y have different numbers of rows - x: ({x.shape[0]}) and y: ({y.shape[0]})"
        )

    if not isinstance(test_size, float):
        raise TypeError(f"test_size must be an float, recieved {type(test_size)}")
    elif test_size < 0 or test_size > 1:
        raise ValueError(f"test_size must be between 0.0 and 1.0. Gave: {test_size}")

    if not isinstance(seed, int):
        raise TypeError(f"seed_num must be an int, recieved {type(seed)}")

    if not isinstance(meta_file, str):
        raise TypeError(f"meta_file must be a str, recieved, {type(meta_file)}")
    elif not os.path.exists(meta_file):
        raise FileNotFoundError(f"file: {meta_file} does not exist")
    elif not os.path.isfile(meta_file):
        raise IsADirectoryError(
            f"meta_file ({meta_file}) points to a directory and not a file "
        )

    if not isinstance(group_name, str):
        raise TypeError(f"group_name must be a str, provided: {type(group_name)}")

    metadata = pd.read_csv(meta_file, index_col=0)

    if group_name not in metadata.columns:
        raise ValueError(
            f"group_name ({group_name}) not present in the meta_data_file ({meta_file})"
        )

    omicLogger.debug("Splitting according to stratification...")

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )

    le = LabelEncoder()
    groups = le.fit_transform(metadata[group_name])

    for train_idx, test_idx in gss.split(x, y, groups):
        if isinstance(x, DataFrame):
            x_train, x_test, y_train, y_test = (
                x.iloc[train_idx, :],
                x.iloc[test_idx, :],
                y.iloc[train_idx, :],
                y.iloc[test_idx, :],
            )
        else:
            x_train, x_test, y_train, y_test = (
                x[train_idx],
                x[test_idx],
                y[train_idx],
                y[test_idx],
            )

    return x_train, x_test, y_train, y_test


def std_split(
    x_full: Union[ndarray, DataFrame],
    y_full: Union[ndarray, DataFrame],
    problem_type: str,
    test_size: float = 0.2,
    seed_num: int = 29292,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Determine the type of train test split to use on the data.

    Parameters
    ----------
    x_full : Union[ndarray, DataFrame]
        The data to be split
    y_full : Union[ndarray, DataFrame]
        The corresponding labels of the data to be split
    problem_type : str
        A str to determin what type of problem it is, must be classification or regression
    test_size : float, optional
        a float to determine the size of the test set, by default 0.2
    seed_num : int, optional
        the seed to control the randomisation, by default 29292

    Returns
    -------
    tuple[ndarray,ndarray,ndarray,ndarray]
        returns the x_train, x_test, y_train and y_test to be used

    Raises
    ------
    TypeError
        is rasied if test_size is not a float
    ValueError
        is raised if test_size is not between 0.0 and 1.0
    TypeError
        Is raised if seed_num is not an in
    TypeError
        is raised if problem_type is not a st
    ValueError
        is raised if problem_type is not classification or regression
    TypeError
        is raised if x_full or y_full is not a ndarray of Dataframe
    ValueError
        is raised if x_full and y_full dont have the same number of rows
    """

    if not isinstance(test_size, float):
        raise TypeError(f"test_size must be an float, recieved {type(test_size)}")
    elif test_size < 0 or test_size > 1:
        raise ValueError(f"test_size must be between 0.0 and 1.0. Gave: {test_size}")

    if not isinstance(seed_num, int):
        raise TypeError(f"seed_num must be an int, recieved {type(seed_num)}")

    if not isinstance(problem_type, str):
        raise TypeError(
            f"problem_type must be an str either {CLASSIFICATION} or {REGRESSION}, recieved {type(test_size)}"
        )
    elif problem_type not in [CLASSIFICATION, REGRESSION]:
        raise ValueError(f"problem_type either {CLASSIFICATION} or {REGRESSION}")

    if not isinstance(x_full, (ndarray, DataFrame)):
        raise TypeError(
            f"x_full must be either a ndarray or a DataFrame. Recieved: {type(x_full)}"
        )

    if not isinstance(y_full, (ndarray, DataFrame)):
        raise TypeError(
            f"y_full must be either a ndarray or a DataFrame. Recieved: {type(y_full)}"
        )

    if x_full.shape[0] != y_full.shape[0]:
        raise ValueError(
            f"x_full and y_full have different numbers of rows - x_full: ({x_full.shape[0]}) and y_full: ({y_full.shape[0]})"
        )

    omicLogger.debug("Split according to standard methods...")

    if problem_type == CLASSIFICATION:
        x_train, x_test, y_train, y_test = train_test_split(
            x_full,
            y_full,
            test_size=test_size,
            random_state=seed_num,
            stratify=y_full,
        )

    elif problem_type == REGRESSION:
        x_train, x_test, y_train, y_test = train_test_split(
            x_full, y_full, test_size=test_size, random_state=seed_num
        )

    return x_train, x_test, y_train, y_test
