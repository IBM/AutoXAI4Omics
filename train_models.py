import argparse
import numpy as np
import pandas as pd
import models
import utils
import plotting
from data_processing import *

##########
import logging
##########

def main(config_dict, config_path):
    '''
    Central function to tie together preprocessing, running the models, and plotting
    '''
    
    # Set the global seed
    np.random.seed(config_dict["seed_num"])
    
    # Create the folders needed
    experiment_folder = utils.create_experiment_folders(config_dict, config_path)
    
    # Set up process logger
    omicLogger = utils.setup_logger(experiment_folder)
    omicLogger.info('Loading data...')
    
    #read the data
    x, y, features_names = load_data(config_dict)
    omicLogger.info('Data Loaded. Splitting data...')
    
    # Split the data in train and test
    x_train, x_test, y_train, y_test = split_data(x, y, config_dict)
    omicLogger.info('Data splitted. Standardising...')
    
    # standardise data
    x_train, SS = standardize_data(x_train) #fit the standardiser to the training data
    x_test = transform_data(x_test,SS) #transform the test data according to the fitted standardiser
    omicLogger.info('Data standardised. Selecting features...')
    
    #implement feature selection if desired
    if config_dict['feature_selection'] is not None:
        x_train, features_names, FS = feat_selection(experiment_folder,x_train, y_train, features_names, config_dict["problem_type"], config_dict['feature_selection'])
        x_test = FS.transform(x_test)
        omicLogger.info('Features selected. Re-combining data...')
    else:
        print("Skipping Feature selection.")
        omicLogger.info('Skipping feature selection. Re-combining data...')
        
    # concatenate both test and train into test
    x = np.concatenate((x_train,x_test))
    y = np.concatenate((y_train,y_test)) #y needs to be re-concatenated as the ordering of x may have been changed in splitting 
    omicLogger.info('Data combined. Defining models...')
    ############ TODO: SAVE DATA TO FILE
    
    
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


    print("----------------------------------------------------------")
    print(f"X data shape: {x.shape}")
    print(f"y data shape: {y.shape}")
    print("Dim train:")
    print(x_train.shape)
    print("Dim test:")
    print(x_test.shape)
    print(f"Number of unique values of target y: {len(np.unique(y))}")
    print("----------------------------------------------------------")


    # Load the models we have pre-defined
    model_dict = models.define_models(config_dict["problem_type"], config_dict["hyper_tuning"])
    omicLogger.info('Models defined. Defining all Scorers...')
    
    #  Define all the scores
    scorer_dict = models.define_scorers(config_dict["problem_type"])
    omicLogger.info('All scorers defined. Extracting chosen scorers...')
    
    #  Select only the scores that you want
    scorer_dict = {k: scorer_dict[k] for k in config_dict["scorer_list"]}
    omicLogger.info('Scorers extracted. Creating results df holder...')
    
    # Create dataframes for results
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    omicLogger.info('Holders created. Begin running models...')
    
    # Run the models
    print("Beginning to run the models")
    models.run_models(
        config_dict=config_dict,
        model_list=config_dict["model_list"],
        model_dict=model_dict,
        df_train=df_train, df_test=df_test,
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        experiment_folder=experiment_folder,
        remove_class=config_dict["remove_classes"],
        merge_class=config_dict["merge_classes"],
        scorer_dict=scorer_dict,
        fit_scorer=config_dict["fit_scorer"],
        hyper_tuning=config_dict["hyper_tuning"],
        hyper_budget=config_dict["hyper_budget"],
        problem_type=config_dict["problem_type"],
        seed_num=config_dict["seed_num"],
        collapse_tax = config_dict["collapse_tax"],  #this is specific to microbiome data
    )
    print("Finished running models!")
    
    omicLogger.info('Models trained. Beggining plotting process...')
    # Plot some graphs
    if config_dict["plot_method"] is not None:
        # See what plots are defined
        plot_dict = plotting.define_plots(config_dict["problem_type"])
        
        omicLogger.info('Plots defined. Begin plotting graphs...')
        # Central func to define the args for the plots
        plotting.plot_graphs(config_dict, experiment_folder, features_names, plot_dict, x, y, x_train, y_train, x_test, y_test, scorer_dict)
    else:
        omicLogger.info('No plots desired.')
        
    omicLogger.info('Process completed.')
        
    ######### TODO: SELECT BEST MODEL

def activate(args):
    parser = argparse.ArgumentParser(description="Explainable AI framework for omics")
    parser = utils.create_parser()
    args = parser.parse_args(args)
    config_path, config_dict = utils.initial_setup(args)

    # This handles pickling issues when cloning for cross-validation
    multiprocessing.set_start_method('spawn', force=True)
    # Run the models
    main(config_dict, config_path)

if __name__ == "__main__":
    # This handles pickling issues when cloning for cross-validation
    import multiprocessing

    multiprocessing.set_start_method('spawn', force=True)
    
    # Load the parser for command line (config files)
    parser = utils.create_parser()
    
    # Get the args
    args = parser.parse_args()
    
    # Do the initial setup
    config_path, config_dict = utils.initial_setup(args)
    
    # Run the models
    main(config_dict, config_path)
