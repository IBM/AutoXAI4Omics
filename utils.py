# --------------------------------------------------------------------------
# Licensed Materials - Property of IBM
#
# (C) Copyright IBM Corp. 2019, 2020
# --------------------------------------------------------------------------

from pathlib import Path
import json
import argparse
import numpy as np
from tensorflow.keras import backend as K
# import scipy.sparse
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import joblib
import pandas as pd
import shap
import models
import plotting
from custom_model import CustomModel, TabAuto
# import calour as ca
from datetime import datetime
import logging
omicLogger = logging.getLogger("OmicLogger")
import yaml

import os
import shutil


def encode_all_categorical(df, include_cols=[], exclude_cols=[]):
    '''
    Encodes all data of type "object" to categorical

    Can provide a list of columns to either include or exclude, depending on the ratio
    '''
    # Loop over the columns (more explicit than select_dtypes)

    for col in df.columns:
        # Check if it's of type object

        if df[col].dtype == "object":
            # Check if it is in our include or (if easier) not in the exclude

            if col in include_cols or col not in exclude_cols:
                # Convert to categorical
                df[col] = df[col].astype('category')
                # Encode using the numerical codes
                df[col] = df[col].cat.codes

def unique_subjects(df):
    '''
    Find the unique subjects by adding the subject number to the study code

    Useful in exploratory data analysis
    '''
    df["Subject"] = df["Subject"].astype(str)
    df["unique_subject"] = df["StudyID"] + "_" + df["Subject"].str[-2:].astype(int).astype(str)
    return df

def remove_classes(class_col, contains="X"):
    # Deprecated! Keeping function here as replacement is specific to Calour - this is specific to Pandas
    return class_col[~class_col.str.contains(contains)]

def check_keys(selection_list, def_dict):
    '''
    Check that each key to be selected exists in the dict that defines them

    Example use case is for our ML models - check that the ones we want to run have been defined
    '''
    for name in selection_list:
        if name not in def_dict:
            raise KeyError(f"{name} is not a defined model in {def_dict}")

def load_model(model_name, model_path):
    '''
    Load a previously saved and trained model. Uses joblib's version of pickle.
    '''
    print("Model path: ")
    print(model_path)
    print("Model ")
    print()

    if model_name in CustomModel.custom_aliases:
        # Remove .pkl here, it will be handled later
        model_path = model_path.replace(".pkl", "")

        try:
            model = CustomModel.custom_aliases[model_name].load_model(model_path)
        except:
            print("The trained model " + model_name + " is not present")
            exit()
    else:
        # Load a previously saved model (using joblib's pickle)
        with open(model_path, 'rb') as f:
            model = joblib.load(f)
    return model

def load_config(config_path):
    '''
    Load a JSON file (general function, but we use it for configs)
    '''
    with open(config_path) as json_file:
        config_dict = json.load(json_file)
    return config_dict

def save_config(experiment_folder, config_path, config_dict):
    '''
    Save the config into the results folder for easy access (storage is cheap right?)
    '''
    # Construct the file name
    fname = experiment_folder / config_path.name
    with open(fname, "w") as outfile:
        json.dump(config_dict, outfile, indent=4)

def check_config(config_dict):
    '''
    Running models can be expensive - let's check that the parameters are valid before wasting time!
    '''
    # Check that all the chosen models are defined
    model_dict = models.define_models(config_dict["problem_type"], config_dict["hyper_tuning"])
    check_keys(config_dict["model_list"], model_dict)
    # Check that the chosen scorers (e.g. accuracy) are defined
    scorer_dict = models.define_scorers(config_dict["problem_type"])
    check_keys(config_dict["scorer_list"], scorer_dict)
    # Check that the fit_scorer is used in scorer_list (otherwise the randomsearch throws an error)
    if config_dict["fit_scorer"] not in config_dict["scorer_list"]:
        raise ValueError(
            f"The fit_scorer must be one of the scorers provided ({config_dict['fit_scorer']} is not in {config_dict['scorer_list']})")
    # Check the plotting params if we're using them
    if config_dict["plot_method"] is not None:
        plot_dict = plotting.define_plots(config_dict["problem_type"])
        check_keys(config_dict["plot_method"], plot_dict)
    # Mac problem with xgboost and openMP
    if "xgboost" in config_dict["model_list"]:
        # Fix for an issue with XGBoost and MacOSX
        import os
        # Check if we're running MacOSX
        try:
            if os.uname()[0] == "Darwin":
                os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        except:
            print("Assuming windows")

def create_experiment_folders(config_dict, config_path):
    '''
    Create the folder for the given config and the relevant subdirectories
    '''
    # Create the folder for this experiment
    experiment_folder = Path(config_dict["save_path"]) / "results" / config_dict["name"]
    # Provide a warning if the folder already exists
    if experiment_folder.is_dir():
        print(f"{experiment_folder} exists - results may be overwritten!")
    experiment_folder.mkdir(parents=True, exist_ok=True)
    # Create the subdirectories
    (experiment_folder / "models").mkdir(exist_ok=True)
    (experiment_folder / "results").mkdir(exist_ok=True)
    if config_dict["plot_method"] is not None:
        (experiment_folder / "graphs").mkdir(exist_ok=True)
    # Save the config in the experiment folder for ease
    save_config(experiment_folder, config_path, config_dict)
    return experiment_folder

def select_explainer(model, model_name, df_train, problem_type):
    '''
    Select the appropriate SHAP explainer for each model
    '''
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

def compute_exemplars_SHAPvalues_withCrossValidation(experiment_folder, config_dict, amp_exp, model, model_name,
                                                     x_train, x_test, y_test, fold_id, pcAgreementLevel=10, save=True):
    feature_names = get_feature_names(amp_exp, config_dict)

    # Convert the data into dataframes to ensure features are displayed
    df_train = pd.DataFrame(data=x_train, columns=feature_names)

    # Select the right explainer from SHAP
    explainer = select_explainer(model, model_name, df_train, config_dict["problem_type"])

    # Get the exemplars  --  to modify to include probability -- get exemplars that have prob > 0.65
    exemplar_X_test = get_exemplars(x_test, y_test, model, config_dict, pcAgreementLevel)
    num_exemplar = exemplar_X_test.shape[0]

    # Save the dataframe with the original exemplars - each row has OTU abundances for each exemplar
    df_exemplars_test = pd.DataFrame(data=exemplar_X_test, columns=feature_names)
    fname_exemplars_test = f"{experiment_folder / 'results' / 'exemplars_abundance'}_{model_name}_{fold_id}"
    df_exemplars_test.to_csv(fname_exemplars_test + '.txt')

    # Compute SHAP values for examplars
    exemplar_shap_values = explainer.shap_values(exemplar_X_test)

    # Classification
    if config_dict["problem_type"] == "classification":

        # For classification there is not difference between data structure returned by SHAP
        exemplars_selected = exemplar_shap_values

        # Try to get the class names
        try:
            class_names = model.classes_.tolist()
        except AttributeError:
            print("Unable to get class names automatically - classes will be encoded")
            class_names = None

    # Regression
    else:
        # Handle Shap saves differently the values for Keras when it's regression
        if model_name == 'mlp_keras':
            exemplars_selected = exemplar_shap_values[0]
        else:
            exemplars_selected = exemplar_shap_values

        # Plot abundance bar plot feature from SHAP
        class_names = []

    features, abundance, abs_shap_values_mean_sorted = compute_average_abundance_top_features(config_dict,
                                                                                              len(feature_names),
                                                                                              model_name, class_names,
                                                                                              amp_exp,
                                                                                              exemplars_selected)

    # Displaying the average percentage %
    abundance = np.asarray(abundance) / 10

    d = {'Features': features,
         'Average Abs Mean SHAP values': abs_shap_values_mean_sorted,
         # 'Average Mean SHAP values':shap_values_mean_sorted,
         'Average abundance': list(abundance)}

    fname = f"{experiment_folder / 'results' / 'all_features_MeanSHAP_Abundance'}_{model_name}_{fold_id}"
    df = pd.DataFrame(d)
    df.to_csv(fname + '.txt')

    # Save exemplars SHAP values
    save_exemplars_SHAP_values(config_dict, experiment_folder, feature_names, model_name, class_names,
                               exemplars_selected, fold_id)

    return num_exemplar

def save_exemplars_SHAP_values(config_dict, experiment_folder, feature_names, model_name, class_names,
                               exemplars_selected, fold_id):
    # Deal with classification differently, classification has shap values for each class
    # Get the SHAP values (global impact) sorted from the highest to the lower (absolute value)
    if config_dict["problem_type"] == "classification":

        # XGBoost for binary classification seems to return the SHAP values only for class 1
        if (model_name == "xgboost" and len(class_names) == 2):
            df_exemplars = pd.DataFrame(data=exemplars_selected, columns=feature_names)
            fname_exemplars = f"{experiment_folder / 'results' / 'exemplars_SHAP_values'}_{model_name}_{fold_id}"
            df_exemplars.to_csv(fname_exemplars + '.txt')

        # When class > 2 (or class > 1 for all the models except XGBoost) SHAP return a list of SHAP value matrices. One for each class.
        else:
            print(type(exemplars_selected))
            print(len(exemplars_selected))
            print(len(exemplars_selected) == len(class_names))

            for i in range(len(exemplars_selected)):
                print('Class: ' + str(i))
                print('Class name: ' + str(class_names[i]))
                df_exemplars = pd.DataFrame(data=exemplars_selected[i], columns=feature_names)
                fname_exemplars = f"{experiment_folder / 'results' / 'exemplars_SHAP_values'}_{model_name}_{class_names[i]}_{i}_{fold_id}"
                df_exemplars.to_csv(fname_exemplars + '.txt')

    # Deal with regression
    else:
        df_exemplars = pd.DataFrame(data=exemplars_selected, columns=feature_names)
        fname_exemplars = f"{experiment_folder / 'results' / 'exemplars_SHAP_values'}_{model_name}_{fold_id}"
        df_exemplars.to_csv(fname_exemplars + '.txt')

def compute_average_abundance_top_features(config_dict, num_top, model_name, class_names, feature_names, data,  shap_values_selected):

    # Get the names of the features
    names = feature_names

    # Create a dataframe to get the average abundance of each feature
    dfMaster = pd.DataFrame(data, columns=names)
    print(dfMaster.head())

    # Order the feature based on SHAP values
    # feature_order = np.argsort(np.sum(np.abs(exemplars_selected), axis=0)) #Sean's version

    # Deal with classification differently, classification has shap values for each class
    # Get the SHAP values (global impact) sorted from the highest to the lower (absolute value)
    if config_dict["problem_type"] == "classification":

        # XGBoost for binary classification seems to return the SHAP values only for class 1
        if (model_name == "xgboost" and len(class_names) == 2):
            feature_order = np.argsort(np.mean(np.abs(shap_values_selected), axis=0))
            shap_values_mean_sorted = np.flip(np.sort(np.mean(np.abs(shap_values_selected), axis=0)))
        # When class > 2 (or class > 1 for all the models except XGBoost) SHAP return a list of SHAP value matrices. One for each class.
        else:
            print(type(shap_values_selected))
            print(len(shap_values_selected))

            shap_values_selected_class = []
            for i in range(len(shap_values_selected)):
                print('Class: ' + str(i))
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

    if(num_top < len(feature_order)):
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
    if config_dict["problem_type"] == "classification":

        print("Classification")
        # Return indices of equal elements between two arrays
        exemplar_indices = np.equal(pred_y, test_y)

    # Regression
    elif config_dict["problem_type"] == "regression":

        print('Regression - Percentage Agreement Level:', pcAgreementLevel)

        if pcAgreementLevel == 0:
            absPcDevArr = np.abs((np.divide(np.subtract(pred_y, test_y), test_y) * 100))
            exemplar_indices = (absPcDevArr == pcAgreementLevel)
        else:
            absPcDevArr = np.abs((np.divide(np.subtract(pred_y, test_y), test_y) * 100))
            exemplar_indices = absPcDevArr < pcAgreementLevel

    # create dataframe for exemplars
    exesToShow = []
    i = 0
    for val in exemplar_indices:
        if val == True:
            exesToShow.append({"idx": i,
                               "testVal": test_y[i],
                               "predVal": pred_y[i]})
        i = i + 1

    # create array with exemplars
    exemplar_X_test = []
    for row in exesToShow:
        exemplar_X_test.append(x_test[int(row['idx'])])
    exemplar_X_test = np.array(exemplar_X_test)

    return exemplar_X_test

def save_explainer(experiment_folder, model_name, explainer):
    save_name = f"{experiment_folder / 'models' / 'explainers' / 'shap'}_{model_name}.pkl"
    with open(save_name, 'wb') as f:
        joblib.dump(explainer, f)

def tidy_tf():
    K.clear_session()

def create_parser():
    parser = argparse.ArgumentParser(description="Microbiome ML Framework")
    # Config argument
    parser.add_argument(
        "-c", "--config", required=True,
        help="Filename of the relevant config file. Automatically selects from configs/ subdirectory."
    )
    return parser

def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])

def initial_setup(args):
    # Construct the config path
    config_path = Path.cwd() / "configs" / args.config
    # print(config_path)
    # config_path = args.config
    config_dict = load_config(config_path)
    # Validate the provided config
    check_config(config_dict)
    # Setup the CustomModel
    # CustomModel.custom_aliases = {k.nickname: k for k in CustomModel.__subclasses__()}
    CustomModel.custom_aliases = {k.nickname: k for k in all_subclasses(CustomModel)}
    return config_path, config_dict

def setup_logger(experiment_folder):
    with open('logging.yml') as file:
        lg_file = yaml.safe_load(file)

    lg_file['handlers']['file']['filename'] =str(experiment_folder/f'AutoOmicLog_{str(int(datetime.timestamp(datetime.utcnow())))}.log')
    logging.config.dictConfig(lg_file)
    
    # formatter = logging.Formatter('%(name)s - %(asctime)s - %(filename)s - %(funcName)s() - %(levelname)s : %(message)s')
    # fh = logging.FileHandler(filename=str(experiment_folder/f'AutoOmicLog_{str(int(datetime.timestamp(datetime.utcnow())))}.log'),mode='a')
    # fh.setFormatter(formatter)
    omicLogger = logging.getLogger("OmicLogger")
    # omicLogger.setLevel(logging.DEBUG)
    # omicLogger.addHandler(fh)
    omicLogger.info('OmicLogger initialised')
    
    return omicLogger

def low_metric_objective(metric):
    """
    Given a metric will return a bool representing wether the objective for the metric is as low as possible (True) or high as possible (False)
    """
    omicLogger.debug("Getting metric objective (High vs Low)...")
    
    objective_low = ['hamming_loss', 'hinge_loss','log_loss','zero_one_loss', 'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'median_absolute_error', 'mean_poisson_deviance', 'mean_gamma_deviance', 
                     'mean_tweedie_deviance']
    objective_high = ['accuracy_score', 'f1_score','jaccard_score','matthews_corrcoef', 'precision_score', 'recall_score','explained_variance_score','r2_score']

    if not ((metric in objective_high) or (metric in objective_low)):
        raise ValueError(f"{metric} not avalable for use")
    else:
        return metric in objective_low
    
def copy_best_content(experiment_folder,best_models,collapse_tax):
    omicLogger.debug("Extracting best model content into unique folder...")
    
    if collapse_tax == None:
        collapse_tax = ''
    
    best = best_models[0]
    alternatives = best_models[1:]
    
    if os.path.exists(experiment_folder/'best_model/'):
        shutil.rmtree(experiment_folder/'best_model/')
     
    os.mkdir(experiment_folder/'best_model/')   
    
    fnames = [os.path.join(path, name) for path, subdirs, files in os.walk(str(experiment_folder)) for name in files]
    sl_fnames = sorted([x for x in fnames if (best in x) and ('.ipynb_checkpoints' not in x)])
    sl_fnames += [x for x in fnames if any(plot in  x for plot in ['barplot','boxplot','feature_selection_accuracy','model_performance']) and ('checkpoint' not in x)]

    for origin in sl_fnames:
        omicLogger.debug(f'copying file {origin}')
        file = os.path.basename(origin)
        target = experiment_folder/f'best_model/{file}'
        shutil.copyfile(origin, target)
        
    if len(alternatives)!=0:
        with open(experiment_folder/'best_model/alternatives.txt', 'w') as fp:
            fp.write('\n'.join(alternatives))
            
    filepath = experiment_folder/f'results/scores_{collapse_tax}_performance_results_testset.csv'
    df = pd.read_csv(filepath)
    df.set_index('model',inplace=True)
    df = df.loc[best]
    df.to_csv(experiment_folder/f'best_model/scores_{collapse_tax}_performance_results_testset.csv')