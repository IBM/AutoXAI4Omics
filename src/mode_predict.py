import os
import pandas as pd
import utils.load
import utils.utils
import logging
import cProfile
from utils.utils import assert_best_model_exists, assert_data_transformers_exists


if __name__ == "__main__":
    """
    Running this script by itself enables for the plots to be made separately from the creation of the models

    Uses the config in the same way as when giving it to run_models.py.
    """

    # init the profiler to time function executions
    pr = cProfile.Profile()
    pr.enable()

    # Do the initial setup
    (
        config_path,
        config_dict,
        experiment_folder,
        omicLogger,
    ) = utils.utils.initial_setup()

    try:
        omicLogger.info("Checking for Trained models")
        model_path = assert_best_model_exists(experiment_folder)

        omicLogger.info("Loading Data...")
        x_to_predict, features_names = utils.load.load_data(config_dict, load_prediction=True)
        x_indexes = x_to_predict.index

        omicLogger.info("Loading data transformers...")
        SS, FS = assert_data_transformers_exists(experiment_folder, config_dict)

        omicLogger.info("Applying trained standardising...")
        x_to_predict = utils.utils.transform_data(x_to_predict, SS)

        if FS is not None:
            omicLogger.info("Applying trained feature selector...")
            x_to_predict = FS.transform(x_to_predict)

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
    utils.utils.prof_to_csv(pr, config_dict)
