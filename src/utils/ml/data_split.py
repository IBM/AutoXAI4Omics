import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, train_test_split
import logging

omicLogger = logging.getLogger("OmicLogger")


################### SPLIT DATA ###################


def split_data(x, y, config_dict):
    """
    Split the data according to the config (i.e normal split or stratify by groups)
    """

    omicLogger.debug("Splitting data...")
    # Split the data in train and test
    if config_dict["ml"]["stratify_by_groups"] == "Y":
        x_train, x_test, y_train, y_test = strat_split(x, y, config_dict)

    else:
        x_train, x_test, y_train, y_test = std_split(x, y, config_dict)

    return x_train, x_test, y_train, y_test


def strat_split(x, y, config_dict):
    """
    split the data according to stratification
    """
    omicLogger.debug("Splitting according to stratification...")

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=config_dict["ml"]["test_size"],
        random_state=config_dict["ml"]["seed_num"],
    )
    # gss = GroupKFold(n_splits=7)

    metadata = pd.read_csv(config_dict["data"]["metadata_file"], index_col=0)
    le = LabelEncoder()
    groups = le.fit_transform(metadata[config_dict["ml"]["groups"]])

    for train_idx, test_idx in gss.split(x, y, groups):
        x_train, x_test, y_train, y_test = (
            x[train_idx],
            x[test_idx],
            y[train_idx],
            y[test_idx],
        )

    return x_train, x_test, y_train, y_test


def std_split(x, y, config_dict):
    """
    Determine the type of train test split to use on the data.
    """
    omicLogger.debug("Split according to standard methods...")

    test_size = config_dict["ml"]["test_size"]
    seed_num = config_dict["ml"]["seed_num"]
    problem_type = config_dict["ml"]["problem_type"]

    if problem_type == "classification":
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, random_state=seed_num, stratify=y
            )
        except:
            print("!!! ERROR: PLEASE SELECT VALID PREDICTION TASK AND TARGET !!!")
            raise

    # Don't stratify for regression (sklearn can't currently handle it with e.g. binning)
    elif problem_type == "regression":
        try:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed_num)
        except:
            print("!!! ERROR: PLEASE SELECT VALID PREDICTION TASK AND TARGET !!!")
            raise

    return x_train, x_test, y_train, y_test
