# --------------------------------------------------------------------------
# Licensed Materials - Property of IBM
#
# (C) Copyright IBM Corp. 2019, 2020
# --------------------------------------------------------------------------

from pathlib import Path
import re
import pdb
import glob
import pickle
import numpy as np
import scipy.sparse
import scipy.stats as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.ticker as ticker
import matplotlib.image as mp_img
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import shap
import eli5
import time
import models
import utils
from custom_model import CustomModel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
import sklearn.metrics as skm
# import imblearn

##########
from data_processing import *
from plotting import *
##########

if __name__ == "__main__":
    '''
    Running this script by itself enables for the plots to be made separately from the creation of the models

    Uses the config in the same way as when giving it to run_models.py.
    '''
    # Load the parser
    parser = utils.create_parser()
    
    # Get the args
    args = parser.parse_args()
    
    # Do the initial setup
    config_path, config_dict = utils.initial_setup(args)

    # Set the global seed
    np.random.seed(config_dict["seed_num"])

    # Create the folders needed
    experiment_folder = utils.create_experiment_folders(config_dict, config_path)
    
    x, y, x_heldout, y_heldout, features_names = load_data(config_dict,load_holdout=True)

    # Split the data in train and test
    x_train, x_test, y_train, y_test = split_data(x, y, config_dict)
    
    # standardise data
    x_train, SS = standardize_data(x_train) #fit the standardiser to the training data
    x_test = transform_data(x_test,SS) #transform the test data according to the fitted standardiser
    x_heldout = transform_data(x_heldout,SS) #transform the holdout data according to the fitted standardiser
    
    #implement feature selection if desired
    if config_dict['feature_selection'] is not None:
        x_train, features_names, FS = feat_selection(experiment_folder,x_train, y_train, features_names, config_dict["problem_type"], config_dict['feature_selection'], save=False)
        x_test = FS.transform(x_test)
        x_heldout = FS.transform(x_heldout)
        
    # concatenate both test and train into test
    x = np.concatenate((x_train,x_test))
    y = np.concatenate((y_train,y_test)) #y needs to be re-concatenated as the ordering of x may have been changed in splitting 
    
    """
    if (config_dict["problem_type"] == "classification"):
        if (config_dict["oversampling"] == "Y"):
            # define oversampling strategy
            oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority')
            # fit and apply the transform
            x_train, y_train = oversample.fit_resample(x_train, y_train)
            print(f"X train data after oversampling shape: {x_train.shape}")
            print(f"y train data after oversampling shape: {y_train.shape}")
    """

    
    # Select only the scorers that we want
    scorer_dict = models.define_scorers(config_dict["problem_type"])
    scorer_dict = {k: scorer_dict[k] for k in config_dict["scorer_list"]}
    
    # See what plots are defined
    plot_dict = define_plots(config_dict["problem_type"])
    
    # Pickling doesn't inherit the self.__class__.__dict__, just self.__dict__
    # So set that up here
    # Other option is to modify cls.__getstate__
    for model_name in config_dict["model_list"]:
        if model_name in CustomModel.custom_aliases:
            CustomModel.custom_aliases[model_name].setup_cls_vars(config_dict, experiment_folder)

    # Create dataframe for performance results
    df_performance_results = pd.DataFrame()

    # Construct the filepath to save the results
    results_folder = experiment_folder / "results"

    if (config_dict["data_type"] == "microbiome"):
        # This is specific to microbiome
        fname = f"scores_{config_dict['collapse_tax']}"
    else:
        fname = "scores_"

    # Remove or merge samples based on target values (for example merging to categories, if classification)
    if config_dict['remove_classes'] is not None:
        fname += "_remove"
    elif config_dict['merge_classes'] is not None:
        fname += "_merge"

    # For each model, load it and then compute performance result
    # Loop over the models
    for model_name in config_dict["model_list"]:
        
        # Load the model
        try:
            model_path = glob.glob(f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}")[0]
        except IndexError:
            print("The trained model " + str('*' + model_name + '*.pkl') + " is not present")
            exit()

        print(f"Plotting barplot for {model_name} using {config_dict['fit_scorer']}")
        model = utils.load_model(model_name, model_path)

        # Evaluate the best model using all the scores and CV
        performance_results_dict = models.evaluate_model(model, config_dict['problem_type'], x_train, y_train, x_heldout, y_heldout)
        
        # Save the results
        df_performance_results, fname_perfResults = models.save_results(
            results_folder, df_performance_results, performance_results_dict,
            model_name, fname, suffix="_performance_results_holdout", save_pkl=False, save_csv=True)

        print(f"{model_name} evaluation on hold out complete! Results saved at {Path(fname_perfResults).parents[0]}")

    # Central func to define the args for the plots
    plot_graphs(config_dict, experiment_folder, features_names, plot_dict, x, y, x_train, y_train, x_heldout, y_heldout, scorer_dict, holdout=True)