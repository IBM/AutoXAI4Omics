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
import shap
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


def select_explainer(model, model_name, df_train, problem_type):
    """
    Select the appropriate SHAP explainer for each model
    """
    # Select the right explainer
    # Note that, for a multi-class (non-binary) problem gradboost cannot use the TreeExplainer
    if model_name in ["xgboost", "rf"]:
        explainer = shap.TreeExplainer(model)
    elif model_name in ["mlp_keras"]:
        explainer = shap.DeepExplainer(model.model, df_train.values)
    elif model_name in ["autolgbm", "autoxgboost"]:
        explainer = shap.TreeExplainer(model.model.model)
    else:
        # KernelExplainer can be very slow, so use their KMeans to speed it up
        # Results are approximate
        df_train_km = shap.kmeans(df_train, 5)
        # For classification we use the predict_proba
        if problem_type == "classification":
            explainer = shap.KernelExplainer(model.predict_proba, df_train_km)
        # Otherwise just use predict
        elif problem_type == "regression":
            explainer = shap.KernelExplainer(model.predict, df_train_km)
    return explainer


# def compute_exemplars_SHAPvalues_withCrossValidation(
#     experiment_folder,
#     config_dict,
#     amp_exp,
#     model,
#     model_name,
#     x_train,
#     x_test,
#     y_test,
#     fold_id,
#     pcAgreementLevel=10,
#     save=True,
# ):
#     feature_names = get_feature_names(amp_exp, config_dict)

#     # Convert the data into dataframes to ensure features are displayed
#     df_train = pd.DataFrame(data=x_train, columns=feature_names)

#     # Select the right explainer from SHAP
#     explainer = select_explainer(model, model_name, df_train, config_dict["ml"]["problem_type"])

#     # Get the exemplars  --  to modify to include probability -- get exemplars that have prob > 0.65
#     exemplar_X_test = get_exemplars(x_test, y_test, model, config_dict, pcAgreementLevel)
#     num_exemplar = exemplar_X_test.shape[0]

#     # Save the dataframe with the original exemplars - each row has OTU abundances for each exemplar
#     df_exemplars_test = pd.DataFrame(data=exemplar_X_test, columns=feature_names)
#     fname_exemplars_test = f"{experiment_folder / 'results' / 'exemplars_abundance'}_{model_name}_{fold_id}"
#     df_exemplars_test.to_csv(fname_exemplars_test + ".txt")

#     # Compute SHAP values for examplars
#     exemplar_shap_values = explainer.shap_values(exemplar_X_test)

#     # Classification
#     if config_dict["ml"]["problem_type"] == "classification":
#         # For classification there is not difference between data structure returned by SHAP
#         exemplars_selected = exemplar_shap_values

#         # Try to get the class names
#         try:
#             class_names = model.classes_.tolist()
#         except AttributeError:
#             print("Unable to get class names automatically - classes will be encoded")
#             class_names = None

#     # Regression
#     else:
#         # Handle Shap saves differently the values for Keras when it's regression
#         if model_name == "mlp_keras":
#             exemplars_selected = exemplar_shap_values[0]
#         else:
#             exemplars_selected = exemplar_shap_values

#         # Plot abundance bar plot feature from SHAP
#         class_names = []

#     features, abundance, abs_shap_values_mean_sorted = compute_average_abundance_top_features(
#         config_dict, len(feature_names), model_name, class_names, amp_exp, exemplars_selected
#     )

#     # Displaying the average percentage %
#     abundance = np.asarray(abundance) / 10

#     d = {
#         "Features": features,
#         "Average Abs Mean SHAP values": abs_shap_values_mean_sorted,
#         # 'Average Mean SHAP values':shap_values_mean_sorted,
#         "Average abundance": list(abundance),
#     }

#     fname = f"{experiment_folder / 'results' / 'all_features_MeanSHAP_Abundance'}_{model_name}_{fold_id}"
#     df = pd.DataFrame(d)
#     df.to_csv(fname + ".txt")

#     # Save exemplars SHAP values
#     save_exemplars_SHAP_values(
#         config_dict, experiment_folder, feature_names, model_name, class_names, exemplars_selected, fold_id
#     )

#     return num_exemplar


def compute_average_abundance_top_features(
    config_dict,
    num_top,
    model_name,
    class_names,
    feature_names,
    data,
    shap_values_selected,
):
    # Get the names of the features
    names = feature_names

    # Create a dataframe to get the average abundance of each feature
    dfMaster = pd.DataFrame(data, columns=names)
    print(dfMaster.head())

    # Order the feature based on SHAP values
    # feature_order = np.argsort(np.sum(np.abs(exemplars_selected), axis=0)) #Sean's version

    # Deal with classification differently, classification has shap values for each class
    # Get the SHAP values (global impact) sorted from the highest to the lower (absolute value)
    if config_dict["ml"]["problem_type"] == "classification":
        # XGBoost for binary classification seems to return the SHAP values only for class 1
        if model_name == "xgboost" and len(class_names) == 2:
            feature_order = np.argsort(np.mean(np.abs(shap_values_selected), axis=0))
            shap_values_mean_sorted = np.flip(np.sort(np.mean(np.abs(shap_values_selected), axis=0)))
        # When class > 2 (or class > 1 for all the models except XGBoost) SHAP return a list of SHAP value matrices.
        # One for each class.
        else:
            print(type(shap_values_selected))
            print(len(shap_values_selected))

            shap_values_selected_class = []
            for i in range(len(shap_values_selected)):
                print("Class: " + str(i))
                shap_values_selected_class.append(np.mean(np.abs(shap_values_selected[i]), axis=0))
            a = np.array(shap_values_selected_class)
            a_mean = np.mean(a, axis=0)
            feature_order = np.argsort(a_mean)
            shap_values_mean_sorted = np.flip(np.sort(a_mean))

    # Deal with regression
    else:
        # Get the SHAP values (global impact) sorted from the highest to the lower (absolute value)
        feature_order = np.argsort(np.mean(np.abs(shap_values_selected), axis=0))
        shap_values_mean_sorted = np.flip(np.sort(np.mean(np.abs(shap_values_selected), axis=0)))

    # In all cases flip feature order anyway to agree with shap_values_mean_sorted
    feature_order = np.flip(feature_order)

    # Select names, average abundance of top features
    top_names = []
    top_averageAbund = []

    if num_top < len(feature_order):
        lim = num_top
    else:
        lim = len(feature_order)

    for j in range(0, lim):
        i = feature_order[j]
        top_names.append(names[i])

        # Get the average of abundance across all the samples not only exemplar
        abund = np.mean(dfMaster[names[i]])
        top_averageAbund.append(abund)

    # Return everything - only SHAP values for the top features
    print("TOP NAMES: ")
    print(top_names)

    print("TOP ABUNDANCE: ")
    print(top_averageAbund)

    return top_names, top_averageAbund, shap_values_mean_sorted[:num_top]


# Get the set of samples for which the prediction are very close to the ground truth
def get_exemplars(x_test, y_test, model, config_dict, pcAgreementLevel):
    # Get the predictions
    pred_y = model.predict(x_test).flatten()

    test_y = y_test
    # test_y = y_test.values - previous version

    # Get the predicted probabilities
    # probs = model.predict_proba(x_test)

    # Create empty array of indices
    exemplar_indices = []

    # Handle classification and regression differently

    # Classification
    if config_dict["ml"]["problem_type"] == "classification":
        print("Classification")
        # Return indices of equal elements between two arrays
        exemplar_indices = np.equal(pred_y, test_y)

    # Regression
    elif config_dict["ml"]["problem_type"] == "regression":
        print("Regression - Percentage Agreement Level:", pcAgreementLevel)

        if pcAgreementLevel == 0:
            absPcDevArr = np.abs((np.divide(np.subtract(pred_y, test_y), test_y) * 100))
            exemplar_indices = absPcDevArr == pcAgreementLevel
        else:
            absPcDevArr = np.abs((np.divide(np.subtract(pred_y, test_y), test_y) * 100))
            exemplar_indices = absPcDevArr < pcAgreementLevel

    # create dataframe for exemplars
    exesToShow = []
    i = 0
    for val in exemplar_indices:
        if val is True:
            exesToShow.append({"idx": i, "testVal": test_y[i], "predVal": pred_y[i]})
        i = i + 1

    # create array with exemplars
    exemplar_X_test = []
    for row in exesToShow:
        exemplar_X_test.append(x_test[int(row["idx"])])
    exemplar_X_test = np.array(exemplar_X_test)

    return exemplar_X_test


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
    score_dict = {
        "acc": "Accuracy",
        "f1": "F1-Score",
        "mean_ae": "Mean Absolute Error",
        "med_ae": "Median Absolute Error",
        "rmse": "Root Mean Squared Error",
        "mean_ape": "Mean Absolute Percentage Error",
        "r2": "R^2",
    }

    if name_type == "model":
        new_name = model_dict[name]
    elif name_type == "score":
        new_name = score_dict[name]
    return new_name


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
