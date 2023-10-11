# process config
# assert that the best model exists (i.e traing has been run)
# load in data to be predicted on
# load data transformers
# transform data
# load model
# predict with model
# save predictions to csv
import os
import glob
import numpy as np
import pandas as pd

import utils.load
import utils.utils
import utils.data_processing as dp
import logging
import joblib
import cProfile
from models.custom_model import CustomModel


def assert_best_model_exists(folder):
    path = folder / "best_model"

    if not os.path.exists(str(path)):
        omicLogger.info("No best model folder detected")
        raise ValueError("No best model folder detected please train model before running prediction")

    found_models = glob.glob(str(path / "*.pkl")) + glob.glob(str(path / "*.h5"))
    if len(found_models) == 0:
        omicLogger.info("No model files detected")
        raise ValueError("No model files detected (.pkl or .h5). Can not perform prediction.")

    omicLogger.info("Best model found ")
    return found_models[0]


def assert_data_transformers_exists(folder, config_dict):
    std = folder / "transformer_std.pkl"

    if not os.path.exists(str(std)):
        omicLogger.info("No data transformer file detected (transformer_std.pkl)")
        raise ValueError("No data transformer file detected (transformer_std.pkl)")
    else:
        with open(std, "rb") as f:
            SS = joblib.load(f)
        omicLogger.info("transformer loaded.")

    if config_dict["ml"]["feature_selection"] is not None:
        fs = folder / "transformer_fs.pkl"
        if not os.path.exists(str(fs)):
            omicLogger.info("No data feature selection file detected (transformer_fs.pkl)")
            raise ValueError("No data feature selection file detected (transformer_fs.pkl)")
        else:
            with open(fs, "rb") as f:
                FS = joblib.load(f)
            omicLogger.info("transformer loaded.")
    else:
        FS = None

    return SS, FS


if __name__ == "__main__":
    """
    Running this script by itself enables for the plots to be made separately from the creation of the models

    Uses the config in the same way as when giving it to run_models.py.
    """
    # Load the parser
    parser = utils.utils.create_parser()

    # Get the args
    args = parser.parse_args()

    # Do the initial setup
    config_path, config_dict = utils.utils.initial_setup(args)

    # init the profiler to time function executions
    pr = cProfile.Profile()
    pr.enable()

    # Set the global seed
    np.random.seed(config_dict["ml"]["seed_num"])

    # Create the folders needed
    experiment_folder = utils.utils.create_experiment_folders(config_dict, config_path)

    # Set up process logger
    omicLogger = utils.utils.setup_logger(experiment_folder)

    try:
        omicLogger.info("Checking for Trained models")
        model_path = assert_best_model_exists(experiment_folder)

        omicLogger.info("Loading Data...")
        x_to_predict, features_names = utils.load.load_data(config_dict, load_prediction=True)
        x_indexes = x_to_predict.index

        omicLogger.info("Loading data transformers...")
        SS, FS = assert_data_transformers_exists(experiment_folder, config_dict)

        omicLogger.info("Applying trained standardising...")
        x_to_predict = dp.transform_data(x_to_predict, SS)

        if FS is not None:
            omicLogger.info("Applying trained feature selector...")
            x_to_predict = FS.transform(x_to_predict)

        for model_name in config_dict["ml"]["model_list"]:
            if model_name in CustomModel.custom_aliases:
                CustomModel.custom_aliases[model_name].setup_cls_vars(config_dict["ml"], experiment_folder)

        model_name = os.path.basename(model_path).split("_")[0]
        omicLogger.debug("Loading model...")
        model = utils.load.load_model(model_name, model_path)

        omicLogger.info("Predicting on data...")
        predictions = model.predict(x_to_predict)

        omicLogger.info("Saving predictions...")
        predictions = pd.DataFrame(predictions, columns=["Prediction"])
        predictions.index = x_indexes
        predictions.index.name = "SampleID"
        predictions.to_csv(
            experiment_folder / f"{config_dict['prediction']['outfile_name']}.csv",
            index=True,
        )

        omicLogger.info("Process completed.")

    except Exception as e:
        omicLogger.error(e, exc_info=True)
        logging.error(e, exc_info=True)
        raise e

    # save time profile information
    pr.disable()
    csv = dp.prof_to_csv(pr)
    with open(
        f"{config_dict['data']['save_path']}results/{config_dict['data']['name']}/time_profile.csv",
        "w+",
    ) as f:
        f.write(csv)
