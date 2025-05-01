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


from datetime import datetime
from models.custom_model import CustomModel
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from typing import Union
from utils.load import load_config
from utils.parser.config_model import ConfigModel
from utils.save import save_config
import argparse
import cProfile
import glob
import io
import joblib
import logging
import numpy as np
import os
import pandas as pd
import pstats
import re
import scipy.sparse
import shutil
import yaml

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
    df["unique_subject"] = (
        df["StudyID"] + "_" + df["Subject"].str[-2:].astype(int).astype(str)
    )
    return df


def remove_classes(class_col, contains="X"):
    # Deprecated! Keeping function here as replacement is specific to Calour - this is specific to Pandas
    return class_col[~class_col.str.contains(contains)]


def create_experiment_folders(config_dict: dict, config_path) -> Path:
    """
    Create the folder for the given config and the relevant subdirectories
    """
    # Create the folder for this experiment
    experiment_folder = (
        Path(config_dict["data"]["save_path"]) / "results" / config_dict["data"]["name"]
    )
    # Provide a warning if the folder already exists
    if experiment_folder.is_dir():
        omicLogger.info(f"{experiment_folder} exists - results may be overwritten!")
    experiment_folder.mkdir(parents=True, exist_ok=True)
    # Create the subdirectories
    (experiment_folder / "models").mkdir(exist_ok=True)
    (experiment_folder / "results").mkdir(exist_ok=True)
    if config_dict["plotting"]["plot_method"] is not None:
        (experiment_folder / "graphs").mkdir(exist_ok=True)

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
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


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
    config_model = ConfigModel(**load_config(config_path))
    config_dict = config_model.model_dump()

    # set the random seed
    set_random_seed(config_dict["ml"]["seed_num"])

    # create folders
    experiment_folder = create_experiment_folders(config_dict, config_path)

    # Save the config in the experiment folder for ease
    save_config(experiment_folder, config_path, config_model.model_dump_json())

    # setup logger
    omicLogger = setup_logger(experiment_folder)

    # Setup the CustomModel
    setup_CustoeModel(config_dict, experiment_folder)

    return config_path, config_dict, experiment_folder, omicLogger


def setup_CustoeModel(config_dict, experiment_folder):
    CustomModel.custom_aliases = {k.nickname: k for k in all_subclasses(CustomModel)}

    for model_name in config_dict["ml"]["model_list"]:
        if model_name in CustomModel.custom_aliases:
            CustomModel.custom_aliases[model_name].setup_cls_vars(
                config_dict["ml"], experiment_folder
            )


def set_random_seed(seed: int):
    np.random.seed(seed)


def setup_logger(experiment_folder):
    with open("logging.yml") as file:
        lg_file = yaml.safe_load(file)

    lg_file["handlers"]["file"]["filename"] = str(
        experiment_folder
        / f"AutoXAI4OmicsLog_{str(int(datetime.timestamp(datetime.utcnow())))}.log"
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

    os.makedirs(experiment_folder / "best_model/")

    fnames = [
        os.path.join(path, name)
        for path, subdirs, files in os.walk(str(experiment_folder))
        for name in files
    ]
    sl_fnames = sorted(
        [x for x in fnames if (best in x) and (".ipynb_checkpoints" not in x)]
    )
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

    filepath = (
        experiment_folder
        / f"results/scores_{collapse_tax}_performance_results_testset.csv"
    )
    df = pd.read_csv(filepath)
    df.set_index("model", inplace=True)
    df = df.loc[best]
    df.to_csv(
        experiment_folder
        / f"best_model/scores_{collapse_tax}_performance_results_testset.csv"
    )


def prof_to_csv(prof: cProfile.Profile, config_dict: dict):
    out_stream = io.StringIO()
    pstats.Stats(prof, stream=out_stream).print_stats()
    result = out_stream.getvalue()
    # chop off header lines
    result = "ncalls" + result.split("ncalls")[-1]
    lines = [",".join(line.rstrip().split(None, 5)) for line in result.split("\n")]
    csv_lines = "\n".join(lines)

    with open(
        f"{config_dict['data']['save_path']}/results/{config_dict['data']['name']}/time_profile.csv",
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

    if name_type == "model":
        res_list = [s for s in re.split("([A-Z][^A-Z]*)", name) if s]
        if all([len(x) > 1 for x in res_list]):
            new_name = " ".join(res_list)
        else:
            new_name = "".join(res_list)

    elif name_type == "score":
        new_name = name.replace("_", " ").capitalize()

    return new_name


def assert_best_model_exists(folder: Path) -> Union[Path, str]:
    path = folder / "best_model"

    if not os.path.exists(str(path)):
        omicLogger.info("No best model folder detected")
        raise ValueError(
            "No best model folder detected please train model before running prediction"
        )

    found_models = glob.glob(str(path / "*.pkl")) + glob.glob(str(path / "*.h5"))
    if len(found_models) == 0:
        omicLogger.info("No model files detected")
        raise ValueError(
            "No model files detected (.pkl or .h5). Can not perform prediction."
        )

    omicLogger.info("Best model found ")
    return found_models[0]


def assert_data_transformers_exists(
    folder: Path, config_dict: dict
) -> tuple[Union[QuantileTransformer, None], Union[Pipeline, None]]:
    """A function to assert that both the standardiser and feature selection transformer exists and if so load them.

    Parameters
    ----------
    folder : Path
        The folder within which to check if the objects exist
    config_dict : dict
        The config dict for the job on which it was trained

    Returns
    -------
    tuple[QuantileTransformer | None, Pipeline | None]
        return the standiser and the pipline object for the feature selection, if either doesn't exist it will return None for that object

    Raises
    ------
    ValueError
        Is raised if the file is not found for either the standardiser or the feature selection object if they were supposed to have existsed
    """

    # If standardisation was done
    if config_dict["ml"]["standardize"]:
        std = folder / "transformer_std.pkl"
        # check if it exists
        if not os.path.exists(str(std)):
            # if not log and raise error
            omicLogger.info("No data transformer file detected (transformer_std.pkl)")
            raise ValueError("No data transformer file detected (transformer_std.pkl)")
        else:
            # if so load for return
            with open(std, "rb") as f:
                SS = joblib.load(f)
            omicLogger.info("transformer loaded.")
    else:
        SS = None

    # if Feature selection was done
    if config_dict["ml"]["feature_selection"] is not None:
        fs = folder / "transformer_fs.pkl"
        # check if it exists
        if not os.path.exists(str(fs)):
            # if nto log and raise error
            omicLogger.info(
                "No data feature selection file detected (transformer_fs.pkl)"
            )
            raise ValueError(
                "No data feature selection file detected (transformer_fs.pkl)"
            )
        else:
            # if so load for return
            with open(fs, "rb") as f:
                FS = joblib.load(f)
            omicLogger.info("transformer loaded.")
    else:
        FS = None

    return SS, FS


def get_model_path(experiment_folder: Path, model_name: str) -> Union[Path, str]:
    try:
        model_path = glob.glob(
            f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}"
        )[0]
    except IndexError as e:
        omicLogger.info(
            "The trained model " + str("*" + model_name + "*.pkl") + " is not present"
        )
        raise e
    return model_path
