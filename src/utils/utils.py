# --------------------------------------------------------------------------
# Licensed Materials - Property of IBM
#
# (C) Copyright IBM Corp. 2019, 2020
# --------------------------------------------------------------------------

import cProfile
import io
from pathlib import Path
import argparse
import pstats
import numpy as np
import pandas as pd
import scipy.sparse
from models.custom_model import CustomModel
from datetime import datetime
import logging
import yaml
import os
import shutil
from utils.load import load_config
from utils.save import save_config
import utils.parsers as parsers

omicLogger = logging.getLogger("OmicLogger")


def encode_all_categorical(df, include_cols=[], exclude_cols=[]):
    """
    Encodes all data of type "object" to categorical

    Can provide a list of columns to either include or exclude, depending on the ratio
    """
    # Loop over the columns (more explicit than select_dtypes)

    for col in df.columns:
        # Check if it's of type object

        if df[col].dtype == "object":
            # Check if it is in our include or (if easier) not in the exclude

            if col in include_cols or col not in exclude_cols:
                # Convert to categorical
                df[col] = df[col].astype("category")
                # Encode using the numerical codes
                df[col] = df[col].cat.codes


def unique_subjects(df):
    """
    Find the unique subjects by adding the subject number to the study code

    Useful in exploratory data analysis
    """
    df["Subject"] = df["Subject"].astype(str)
    df["unique_subject"] = df["StudyID"] + "_" + df["Subject"].str[-2:].astype(int).astype(str)
    return df


def remove_classes(class_col, contains="X"):
    # Deprecated! Keeping function here as replacement is specific to Calour - this is specific to Pandas
    return class_col[~class_col.str.contains(contains)]


def create_experiment_folders(config_dict, config_path):
    """
    Create the folder for the given config and the relevant subdirectories
    """
    # Create the folder for this experiment
    experiment_folder = Path(config_dict["data"]["save_path"]) / "results" / config_dict["data"]["name"]
    # Provide a warning if the folder already exists
    if experiment_folder.is_dir():
        print(f"{experiment_folder} exists - results may be overwritten!")
    experiment_folder.mkdir(parents=True, exist_ok=True)
    # Create the subdirectories
    (experiment_folder / "models").mkdir(exist_ok=True)
    (experiment_folder / "results").mkdir(exist_ok=True)
    if config_dict["plotting"]["plot_method"] is not None:
        (experiment_folder / "graphs").mkdir(exist_ok=True)
    # Save the config in the experiment folder for ease
    save_config(experiment_folder, config_path, config_dict)
    return experiment_folder


# Get the set of samples for which the prediction are very close to the ground truth
def create_cli_parser():
    parser = argparse.ArgumentParser(description="Microbiome ML Framework")
    # Config argument
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Filename of the relevant config file. Automatically selects from configs/ subdirectory.",
    )
    return parser


def all_subclasses(cls):
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])


def get_config_path_from_cli():
    # Load the parser for command line (config files)
    cli_parser = create_cli_parser()

    # Get the args
    cli_args = cli_parser.parse_args()

    # Construct the config path
    config_path = Path.cwd() / "configs" / cli_args.config
    return config_path


def initial_setup():
    # get the path to the config from cli
    config_path = get_config_path_from_cli()
    # load and parse the config located at the path
    config_dict = parsers.parse_config(load_config(config_path))

    # Setup the CustomModel
    CustomModel.custom_aliases = {k.nickname: k for k in all_subclasses(CustomModel)}

    # set the random seed
    set_random_seed(config_dict["ml"]["seed_num"])

    # create folders
    experiment_folder = create_experiment_folders(config_dict, config_path)

    # setup logger
    omicLogger = setup_logger(experiment_folder)

    return config_path, config_dict, experiment_folder, omicLogger


def set_random_seed(seed: int):
    np.random.seed(seed)


def setup_logger(experiment_folder):
    with open("logging.yml") as file:
        lg_file = yaml.safe_load(file)

    lg_file["handlers"]["file"]["filename"] = str(
        experiment_folder / f"AutoOmicLog_{str(int(datetime.timestamp(datetime.utcnow())))}.log"
    )
    logging.config.dictConfig(lg_file)
    omicLogger = logging.getLogger("OmicLogger")
    # omicLogger.setLevel(logging.DEBUG)
    omicLogger.info("OmicLogger initialised")

    return omicLogger


def copy_best_content(experiment_folder, best_models, collapse_tax):
    omicLogger.debug("Extracting best model content into unique folder...")

    if collapse_tax is None:
        collapse_tax = ""

    best = best_models[0]
    alternatives = best_models[1:]

    if os.path.exists(experiment_folder / "best_model/"):
        shutil.rmtree(experiment_folder / "best_model/")

    os.mkdir(experiment_folder / "best_model/")

    fnames = [os.path.join(path, name) for path, subdirs, files in os.walk(str(experiment_folder)) for name in files]
    sl_fnames = sorted([x for x in fnames if (best in x) and (".ipynb_checkpoints" not in x)])
    sl_fnames += [
        x
        for x in fnames
        if any(
            plot in x
            for plot in [
                "barplot",
                "boxplot",
                "feature_selection_accuracy",
                "model_performance",
            ]
        )
        and ("checkpoint" not in x)
    ]

    for origin in sl_fnames:
        omicLogger.debug(f"copying file {origin}")
        file = os.path.basename(origin)
        target = experiment_folder / f"best_model/{file}"
        shutil.copyfile(origin, target)

    if len(alternatives) != 0:
        with open(experiment_folder / "best_model/alternatives.txt", "w") as fp:
            fp.write("\n".join(alternatives))

    filepath = experiment_folder / f"results/scores_{collapse_tax}_performance_results_testset.csv"
    df = pd.read_csv(filepath)
    df.set_index("model", inplace=True)
    df = df.loc[best]
    df.to_csv(experiment_folder / f"best_model/scores_{collapse_tax}_performance_results_testset.csv")


def prof_to_csv(prof: cProfile.Profile, config_dict: dict):
    out_stream = io.StringIO()
    pstats.Stats(prof, stream=out_stream).print_stats()
    result = out_stream.getvalue()
    # chop off header lines
    result = "ncalls" + result.split("ncalls")[-1]
    lines = [",".join(line.rstrip().split(None, 5)) for line in result.split("\n")]
    csv_lines = "\n".join(lines)

    with open(
        f"{config_dict['data']['save_path']}results/{config_dict['data']['name']}/time_profile.csv",
        "w+",
    ) as f:
        f.write(csv_lines)


def transform_data(data, transformer):
    omicLogger.debug("Transforming given data according to given transformer...")

    if scipy.sparse.issparse(data):
        data = data.todense()
    else:
        data = data

    try:
        data = transformer.transform(data)
        return data
    except Exception:
        raise TypeError("Supplied transformer does not have the transform method")


def pretty_names(name, name_type):
    omicLogger.debug("Fetching pretty names...")
    model_dict = {
        "rf": "RF",
        "mlp_keras": "DeepNN",
        "tab_auto": "TabAuto",
        "autokeras": "A/Keras",
        "fixedkeras": "Keras",
        "autolgbm": "A/LGBM",
        "autoxgboost": "A/XGB",
        "autosklearn": "A/SKL",
        "autogluon": "A/Gluon",
        "svm": "SVM",
        "knn": "KNN",
        "xgboost": "XGBoost",
        "adaboost": "AdaBoost",
    }
    # score_dict = {
    #     "acc": "Accuracy",
    #     "f1": "F1-Score",
    #     "mean_ae": "Mean Absolute Error",
    #     "med_ae": "Median Absolute Error",
    #     "rmse": "Root Mean Squared Error",
    #     "mean_ape": "Mean Absolute Percentage Error",
    #     "r2": "R^2",
    # }

    if name_type == "model":
        new_name = model_dict[name]
    elif name_type == "score":
        new_name = name.replace("_", " ").capitalize()
        # new_name = score_dict[name]
    return new_name
