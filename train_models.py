
import argparse
import numpy as np
import pandas as pd
import models
import utils
from sklearn.preprocessing import StandardScaler
import scipy.sparse
import plotting
import subprocess

""" Standardize the input X using Standard Scaler"""
def standardize_data(data):

    if scipy.sparse.issparse(data):
        data = data.todense()
    else:
        data = data
    data = StandardScaler().fit_transform(data)
    return data

""" Read the input files and return X, y (target) and the feature_names"""
def get_data(path_file, target, metadata_path):

    # Read the data
    data = pd.read_csv(path_file, index_col=0)
    print("Data dimension: "+str(data.shape))

    # Check if the target is in a separate file or in the same data
    if(metadata_path == ""):
        y = data[target].values
        data_notarget = data.drop(target, axis=1)

    else: # it assumes the data does not contain the target column
        # Read the metadata file
        metadata = pd.read_csv(metadata_path, index_col=0)
        y = metadata[target].values
        data_notarget = data

    features_names = data_notarget.columns
    x = data_notarget.values

    # Scale x
    x = standardize_data(x)

    # Check the data and labels are the right size
    assert len(x) == len(y)

    return x,y,features_names

def get_data_microbiome(path_file, metadata_path, config_dict):
    '''
    Load and process the data
    '''

    # Use calour to create an experiment
    print("Path file: " +path_file)
    print("Metadata file: " +metadata_path)
    amp_exp = utils.create_microbiome_calourexp(path_file, metadata_path)

    print("")
    print("")
    print("")
    print("***** Preprocessing microbiome data *******")

    print(f"Original data dimension: {amp_exp.data.shape}")
    # Use calour to filter the data

    amp_exp = utils.filter_biom(amp_exp, collapse_tax=config_dict["collapse_tax"])
    print(f"After filtering contaminant, collapsing at genus and filtering by abundance: {amp_exp.data.shape}")

    # Filter any data that needs it
    if config_dict["filter_samples"] is not None:
        amp_exp = utils.filter_samples(amp_exp, config_dict["filter_samples"])

    # Modify the classes if need be
    amp_exp = utils.modify_classes(
        amp_exp,
        config_dict["target"],
        remove_class=config_dict["remove_classes"],
        merge_by=config_dict["merge_classes"]
    )

    print(f"After filtering samples: {amp_exp.data.shape}")

    print("Save experiment after filtering with name exp_filtered")
    amp_exp.save('exp_filtered')
    print("****************************************************")
    print("")
    print("")
    print("")

    # Prepare data (load and normalize)
    x = utils.prepare_data(amp_exp)
    try:
        # Select the labels
        y = utils.select_class_col(
            amp_exp,
            encoding=config_dict["encoding"], #from Cameron
            name=config_dict["target"]
        )
    except:
        print("!!! ERROR: PLEASE SELECT TARGET TO PREDICT FROM METADATA FILE !!!")

    features_names = utils.get_feature_names_calourexp(amp_exp, config_dict)

    # print(f"Class col:\n{y}")
    # Check the data and labels are the right size
    assert len(x) == len(y)

    return  x, y, features_names

def get_data_gene_expression(path_file, metadata_path, config_dict):
    '''
    Load and process the data
    '''

    # Use calour to create an experiment
    print("Path file: " +path_file)
    print("Metadata file: " +metadata_path)

    print("")
    print("")
    print("")
    print("***** Preprocessing gene expression data *******")

    # add the input file parameter
    strcommand = "--expressionfile "+ path_file + " "

    # add the expression type parameter that is required
    if config_dict["expression_type"] is not None:
        expression_type = config_dict["expression_type"]
        print(expression_type)
    else:
        expression_type = "OTHERS"
    strcommand = strcommand+"--expressiontype "+expression_type+ " "

    # add the filter_samples parameter that is optional
    if config_dict["filter_sample"] is not None:
        filter_samples = config_dict["filter_sample"]
        print(filter_samples)
        strcommand = strcommand + "--Filtersamples " + str(filter_samples) + " "

    # add the filter_genes parameter that is optional
    if config_dict["filter_genes"] is not None:
        filter_genes = config_dict["filter_genes"][0] +" "+config_dict["filter_genes"][1]
        print(filter_genes)
        strcommand = strcommand+"--Filtergenes "+filter_genes+" "

    # add the output file name that is required
    if config_dict["output_file_ge"] is not None:
        output_file = config_dict["output_file_ge"]
        print(output_file)
    else:
        output_file = "processed_gene_expression_data"
    strcommand = strcommand+"--output "+output_file
    print(strcommand)

    python_command = "python AoT_gene_expression_pre_processing.py "+strcommand
    print(python_command)
    subprocess.call(python_command, shell=True)
    x,y, feature_names = get_data(config_dict["output_file_ge"], config_dict["target"], metadata_path)

    return x,y, feature_names

'''
    Central function to tie together preprocessing, running the models, and plotting
'''
def main(config_dict, config_path):

    # Set the global seed
    np.random.seed(config_dict["seed_num"])

    if(config_dict["data_type"]=="microbiome"):
        # This reads and preprocesses microbiome data using calour library -- it would be better to change this preprocessing so that it is not dependent from calour
        x,y,features_names = get_data_microbiome(config_dict["file_path"], config_dict["metadata_file"], config_dict)
    elif(config_dict["data_type"] == "gene_expression"):
        # This reads and preprocesses microbiome data using calour library -- it would be better to change this preprocessing so that it is not dependent from calour
        x, y, features_names = get_data_gene_expression(config_dict["file_path"], config_dict["metadata_file"], config_dict)
    else:
        # At the moment for all the other data types, for example metabolomics, we have not implemented preprocessing except for standardisation with StandardScaler()
        x, y, features_names = get_data(config_dict["file_path"], config_dict["target"], config_dict["metadata_file"])

    # Split the data in train and test
    x_train, x_test, y_train, y_test = models.split_data(
        x, y, config_dict["test_size"], config_dict["seed_num"],
        config_dict["problem_type"]
    )

    print("----------------------------------------------------------")
    print(f"X data shape: {x.shape}")
    print(f"y data shape: {y.shape}")
    print(f"Number of unique values of target y: {len(np.unique(y))}")
    print("----------------------------------------------------------")

    # Create the folders needed
    experiment_folder = utils.create_experiment_folders(config_dict, config_path)

    # Load the models we have pre-defined
    model_dict = models.define_models(config_dict["problem_type"], config_dict["hyper_tuning"])

    #  Define all the scores
    scorer_dict = models.define_scorers(config_dict["problem_type"])

    #  Select only the scores that you want
    scorer_dict = {k: scorer_dict[k] for k in config_dict["scorer_list"]}

    # Create dataframes for results

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

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

    # Plot some graphs
    if config_dict["plot_method"] is not None:
        # See what plots are defined
        plot_dict = plotting.define_plots(config_dict["problem_type"])
        # Central func to define the args for the plots
        plotting.plot_graphs(config_dict, experiment_folder, features_names, plot_dict, x, y, x_train, y_train, x_test, y_test, scorer_dict)

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