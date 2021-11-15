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
import train_models
import utils
from custom_model import CustomModel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
import sklearn.metrics as skm


# import imblearn


def define_plots(problem_type):
    '''
    Define the plots for each problem type. This needs to be maintained manually.
    '''
    # Some plots can only be done for a certain type of ML
    # Check here that the ones given are valid
    if problem_type == "classification":
        plot_dict = {
            "barplot_scorer": barplot_scorer,
            "boxplot_scorer": boxplot_scorer_cv,
            "boxplot_scorer_cv_groupby": boxplot_scorer_cv_groupby,
            "conf_matrix": conf_matrix_plot,
            "shap_plots": shap_plots,
            "shap_force_plots": shap_force_plots,
            "permut_imp_test": permut_importance
        }
    elif problem_type == "regression":
        plot_dict = {
            "barplot_scorer": barplot_scorer,
            "boxplot_scorer": boxplot_scorer_cv,
            "boxplot_scorer_cv_groupby": boxplot_scorer_cv_groupby,
            "shap_plots": shap_plots,
            "shap_force_plots": shap_force_plots,
            "corr": correlation_plot,
            "hist": distribution_hist,
            "hist_overlapped": histograms,
            "joint": joint_plot,
            "joint_dens": joint_plot,
            "permut_imp_test": permut_importance
        }
    return plot_dict


def plot_graphs(config_dict, experiment_folder, feature_names, plot_dict, x, y, x_train, y_train, x_test, y_test,
                scorer_dict):
    '''
    Plot graphs as specified by the config. Each plot function is handled separately to be explicit (at the cost of length and maintenance).
    Here you can customize whether you want to graph on train or test based on what arguments are given for the data and labels.
    '''

    # Loop over every plot method we're using
    for plot_method in config_dict["plot_method"]:
        plot_func = plot_dict[plot_method]
        print(plot_method)
        # Hand-crafted passing the arguments in because over-engineering
        # Don't judge me (I'm not a big **kwargs fan)
        if plot_method == "barplot_scorer":
            plot_func(experiment_folder, config_dict, scorer_dict, x_test, y_test)
        elif plot_method == "boxplot_scorer":
            plot_func(experiment_folder, config_dict, scorer_dict, x_train, y_train)
        elif plot_method == "boxplot_scorer_cv_groupby":
            plot_func(experiment_folder, config_dict, scorer_dict, x, y)
        elif plot_method == "conf_matrix":
            plot_func(experiment_folder, config_dict, x_test, y_test, normalize=False)
        elif plot_method == "corr":
            plot_func(experiment_folder, config_dict, x_test, y_test, config_dict["target"])
        elif plot_method == "hist":
            plot_func(experiment_folder, config_dict, x_test, y_test, config_dict["target"])
        elif plot_method == "hist_overlapped":
            plot_func(experiment_folder, config_dict, x_test, y_test, config_dict["target"])
        elif plot_method == "joint":
            plot_func(experiment_folder, config_dict, x_test, y_test, config_dict["target"])
        elif plot_method == "joint_dens":
            plot_func(experiment_folder, config_dict, x_test, y_test, config_dict["target"], kind="kde")
        elif plot_method == "permut_imp_test":
            plot_func(experiment_folder, config_dict, scorer_dict, feature_names, x_test, y_test,
                      config_dict["top_feats_permImp"], cv='prefit')
        elif plot_method == "shap_plots":
            plot_func(experiment_folder, config_dict, feature_names, x, x_test, y_test, x_train,
                      config_dict["top_feats_shap"])

        """elif plot_method == "shap_force_plots":
            plot_func(experiment_folder, config_dict, x_test, y_test, feature_names, x, y, x_train, data_forexplanations="all",
                             top_exemplars=0.4, save=True)
        elif plot_method == "permut_imp_alldata":
            plot_func(experiment_folder, config_dict, scorer_dict, feature_names, x, y, config_dict["top_feats_permImp"], cv='prefit')
        elif plot_method == "permut_imp_train":
            plot_func(experiment_folder, config_dict, scorer_dict, feature_names, x_train, y_train, config_dict["top_feats_permImp"], cv='prefit')
        elif plot_method == "permut_imp_5cv":
            plot_func(experiment_folder, config_dict, scorer_dict, feature_names, x, y, config_dict["top_feats_permImp"], cv=5)"""

    # Clear everything
    plt.clf()
    plt.close()
    # Clear keras and TF sessions/graphs etc.
    utils.tidy_tf()


def summary_SHAPdotplot_perclass(experiment_folder, class_names, model_name, feature_names, num_top, exemplar_X_test,
                                 exemplars_selected):
    if (model_name == 'xgboost' and len(class_names) == 2):
        print('Shape exemplars_selected: ' + str(exemplars_selected.shape))
        class_name = class_names[1]
        print('Class: ' + str(class_name))
        fname = f"{experiment_folder / 'graphs' / 'summary_SHAPdotplot_perclass'}_{model_name}_{class_name}_{'holdout'}"

        # Plot shap bar plot
        shap.summary_plot(
            exemplars_selected,
            exemplar_X_test,
            plot_type='dot',
            color_bar='000',
            max_display=num_top,
            feature_names=feature_names,
            show=False
        )
        fig = plt.gcf()
        save_fig(fig, fname)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()

    else:
        for i in range(len(class_names)):
            class_name = class_names[i]
            print('Class: ' + str(class_name))
            print('i:' + str(i))

            print('Length exemplars_selected: ' + str(len(exemplars_selected)))
            print('Type exemplars_selected: ' + str(type(exemplars_selected)))

            fname = f"{experiment_folder / 'graphs' / 'summary_SHAPdotplot_perclass'}_{model_name}_{class_name}_{'holdout'}_{i}"

            # Plot shap bar plot
            shap.summary_plot(
                exemplars_selected[i],
                exemplar_X_test,
                plot_type='dot',
                color_bar='000',
                max_display=num_top,
                feature_names=feature_names,
                show=False
            )
            my_cmap = plt.get_cmap('viridis')

            # Change the colormap of the artists
            for fc in plt.gcf().get_children():
                for fcc in fc.get_children():
                    if hasattr(fcc, "set_cmap"):
                        fcc.set_cmap(my_cmap)
            fig = plt.gcf()
            save_fig(fig, fname)
            plt.draw()
            plt.tight_layout()
            plt.pause(0.001)
            time.sleep(2)
            # Close the figure to ensure we start anew
            plt.clf()
            plt.close()

    plt.clf()
    plt.close()


def save_fig(fig, fname, dpi=200, fig_format="png"):
    print(f"Save location: {fname}.{fig_format}")
    fig.savefig(
        f"{fname}.{fig_format}",
        dpi=dpi,
        format=fig_format,
        bbox_inches='tight',
        transparent=False
    )


def create_fig(nrows=1, ncols=1, figsize=None):
    '''
    Universal call to subplots to allow consistent specification of e.g. figsize
    '''
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, ax


def pretty_names(name, name_type):
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
        "adaboost": "AdaBoost"
    }
    score_dict = {
        "acc": "Accuracy",
        "f1": "F1-Score",
        "mean_ae": "Mean Absolute Error",
        "med_ae": "Median Absolute Error",
        "rmse": "Root Mean Squared Error"
    }

    if name_type == "model":
        new_name = model_dict[name]
    elif name_type == "score":
        new_name = score_dict[name]
    return new_name


def boxplot_scorer_cv(experiment_folder, config_dict, scorer_dict, data, true_labels, nsplits=5, save=True):
    '''
    Create a graph of boxplots for all models in the folder, using the specified fit_scorer from the config.

    By default this uses a 5-fold stratified cross validation. Also it saves the list of SHAP values for each of the exemplars of
    each fold
    '''
    # Create the plot objects
    fig, ax = plt.subplots()
    # Container for the scores
    all_scores = []
    print(f"Size of data for boxplot: {data.shape}")
    # Create the fold object for CV
    if config_dict["problem_type"] == "classification":
        fold_obj = StratifiedKFold(
            n_splits=nsplits,
            shuffle=True,
            random_state=config_dict["seed_num"]
        )
    elif config_dict["problem_type"] == "regression":
        fold_obj = KFold(
            n_splits=nsplits,
            shuffle=True,
            random_state=config_dict["seed_num"]
        )
    # Loop over the models
    for model_name in config_dict["model_list"]:
        # Load the model if trained
        try:
            model_path = glob.glob(f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}")[0]
        except IndexError:
            print("The trained model " + str('*' + model_name + '*.pkl') + " is not present")
            exit()

        print(f"Plotting boxplot for {model_name} using {config_dict['fit_scorer']}")
        # Select the scorer
        scorer_func = scorer_dict[config_dict['fit_scorer']]
        # Container for scores for this cross-val for this model
        scores = []
        num_testsamples_list = []
        num_fold = 0

        for train_idx, test_idx in fold_obj.split(data, true_labels):
            print(f"{model_name}, fold {num_fold}")
            num_fold += 1
            # Load the model
            model = utils.load_model(model_name, model_path)
            # Handle the custom model
            if isinstance(model, tuple(CustomModel.__subclasses__())):
                # Remove the test data to avoid any saving
                if model.data_test is not None:
                    model.data_test = None
                if model.labels_test is not None:
                    model.labels_test = None
            # For CustomModels, we do not want to save the model
            if model_name in CustomModel.custom_aliases:
                model.fit(data[train_idx], true_labels[train_idx], save_best=False)
            # Otherwise fit the model as normal
            else:
                model.fit(data[train_idx], true_labels[train_idx])
            # Calculate the score
            # Need to take the absolute value because of the make_scorer sklearn convention
            score = np.abs(scorer_func(
                model,
                data[test_idx],
                true_labels[test_idx]
            ))
            num_testsamples = len(true_labels[test_idx])
            # Add the scores
            scores.append(score)
            num_testsamples_list.append(num_testsamples)

            # Clear keras and TF sessions/graphs etc.
            # utils.tidy_tf()

        # Maintain the total list
        all_scores.append(scores)
        # Save CV results
        d = {'Scores CV': scores, 'Dim test': num_testsamples_list}
        fname = f"{experiment_folder / 'results' / 'scores_CV'}_{model_name}_{num_fold}_{'holdout'}"
        df = pd.DataFrame(d)
        df.to_csv(fname + '.csv')

    pretty_model_names = [pretty_names(name, "model") for name in config_dict["model_list"]]

    # Make a dataframe
    """
    df_cv_scores = pd.DataFrame(all_scores, columns=pretty_model_names)
    fname_all_cv = f"{experiment_folder / 'results' / 'scores_CV_allmodels'}"
    df_cv_scores.to_csv(fname_all_cv+".csv")
    print(df_cv_scores)
    """

    # Make the boxplot
    sns.boxplot(x=pretty_model_names, y=all_scores, ax=ax, width=0.4)
    # Format the graph
    # ax.set_xticklabels(pretty_names(config_dict["model_list"], "score"))
    ax.set_xlabel("ML Methods")

    fig = plt.gcf()
    # Save the graph
    if save:
        fname = f"{experiment_folder / 'graphs' / 'boxplot'}_{config_dict['fit_scorer']}_{'holdout'}"
        save_fig(fig, fname)

    # Close the figure to ensure we start anew
    plt.clf()
    plt.close()


def boxplot_scorer_cv_groupby(experiment_folder, config_dict, scorer_dict, data, true_labels, save=True):
    '''
    Create a graph of boxplots for all models in the folder, using the specified fit_scorer from the config.
    '''
    # Create the plot objects
    fig, ax = plt.subplots()
    # Container for the scores
    all_scores = []
    print(f"Size of data for boxplot: {data.shape}")

    # GroupBy Subject ID -- TO FINISH CODING
    metadata = pd.read_csv(config_dict["metadata_file"], index_col=0)
    le = LabelEncoder()
    groups = le.fit_transform(metadata[config_dict["groups"]])

    fold_obj = GroupShuffleSplit(n_splits=5, test_size=config_dict["test_size"], random_state=config_dict["seed_num"])

    # fold_obj = GroupKFold(n_splits=5)

    # Loop over the models
    for model_name in config_dict["model_list"]:
        # Load the model if trained
        try:
            model_path = glob.glob(f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}")[0]
        except IndexError:
            print("The trained model " + str('*' + model_name + '*.pkl') + " is not present")
            exit()

        print(f"Plotting boxplot for {model_name} using {config_dict['fit_scorer']} - Grouped By " + config_dict[
            "groups"])
        # Select the scorer
        scorer_func = scorer_dict[config_dict['fit_scorer']]
        # Container for scores for this cross-val for this model
        scores = []
        num_testsamples_list = []
        num_fold = 0

        for train_idx, test_idx in fold_obj.split(data, true_labels, groups):
            print(f"{model_name}, fold {num_fold}")
            num_fold += 1
            # Load the model
            model = utils.load_model(model_name, model_path)
            # Handle the custom model
            if isinstance(model, tuple(CustomModel.__subclasses__())):
                # Remove the test data to avoid any saving
                if model.data_test is not None:
                    model.data_test = None
                if model.labels_test is not None:
                    model.labels_test = None
            """
            # Option of oversampling of the training dataset for classification
            if (config_dict["problem_type"] == "classification"):
                if (config_dict["oversampling"] == "Y"):
                    # define oversampling strategy
                    oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority')
                    # fit and apply the transform
                    x_train, y_train = oversample.fit_resample(x_train, y_train)
                    print(f"X train data after oversampling shape: {x_train.shape}")
                    print(f"y train data after oversampling shape: {y_train.shape}")

            """
            """
            if model_name in CustomModel.custom_aliases:
                if (model_name == "autolgbm"):
                    print("It's autolgbm don't do anything")
                else:
                    model.fit(data[train_idx], true_labels[train_idx], save_best=False)
                # Otherwise fit the model as normal
            else:
                model.fit(data[train_idx], true_labels[train_idx])
            """

            model.fit(data[train_idx], true_labels[train_idx])

            # Calculate the score
            # Need to take the absolute value because of the make_scorer sklearn convention
            score = np.abs(scorer_func(
                model,
                data[test_idx],
                true_labels[test_idx]
            ))
            num_testsamples = len(true_labels[test_idx])
            # Add the scores
            scores.append(score)
            num_testsamples_list.append(num_testsamples)

        # Maintain the total list
        all_scores.append(scores)
        # Save CV results
        d = {'Scores CV': scores, 'Dim test': num_testsamples_list}
        fname = f"{experiment_folder / 'results' / 'GroupShuffleSplit_CV'}_{model_name}_{num_fold}_{'holdout'}"
        df = pd.DataFrame(d)
        df.to_csv(fname + '.csv')

    pretty_model_names = [pretty_names(name, "model") for name in config_dict["model_list"]]

    # Make the boxplot
    sns.boxplot(x=pretty_model_names, y=all_scores, ax=ax, width=0.4)
    # Format the graph
    # ax.set_xticklabels(pretty_names(config_dict["model_list"], "score"))
    ax.set_xlabel("ML Methods")

    fig = plt.gcf()
    # Save the graph
    if save:
        fname = f"{experiment_folder / 'graphs' / 'boxplot_GroupShuffleSplit_CV'}_{config_dict['fit_scorer']}_{'holdout'}"
        save_fig(fig, fname)

    # Close the figure to ensure we start anew
    plt.clf()
    plt.close()


def barplot_scorer(experiment_folder, config_dict, scorer_dict, data, true_labels, save=True):
    '''
    Create a barplot for all models in the folder using the fit_scorer from the config.
    '''
    # Create the plot objects
    fig, ax = plt.subplots()
    # Container for the scores
    all_scores = []
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
        # Get our single score
        score = np.abs(scorer_dict[config_dict['fit_scorer']](model, data, true_labels))
        all_scores.append(score)
        # Clear keras and TF sessions/graphs etc.
        utils.tidy_tf()
    pretty_model_names = [pretty_names(name, "model") for name in config_dict["model_list"]]
    # Make the barplot
    sns.barplot(x=pretty_model_names, y=all_scores, ax=ax)
    # ax.set_xticklabels(pretty_model_names)
    ax.set_ylabel(pretty_names(config_dict["fit_scorer"], "score"))
    ax.set_xlabel("Model")
    ax.set_title(f"Performance on test data")
    if save:
        fname = f"{experiment_folder / 'graphs' / 'barplot'}_{config_dict['fit_scorer']}_{'holdout'}"
        save_fig(fig, fname)

    plt.draw()
    plt.tight_layout()
    plt.pause(0.001)
    time.sleep(2)
    # Close the figure to ensure we start anew
    plt.clf()
    plt.close()


def shap_summary_plot(experiment_folder, config_dict, x_test, feature_names, shap_dict, save=True):
    '''
    A wrapper to prepare the data and models for the SHAP summary plot
    '''
    # Convert the data into dataframes to ensure features are displayed
    df_test = pd.DataFrame(data=x_test, columns=feature_names)
    # Get the model paths
    for model_name in config_dict["model_list"]:

        try:
            model_path = glob.glob(f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}")[0]
        except IndexError:
            print("The trained model " + str('*' + model_name + '*.pkl') + " is not present")
            exit()

        print(f"Plotting SHAP for {model_name}")
        model = utils.load_model(model_name, model_path)
        # Define the figure object
        fig, ax = plt.subplots()
        # Select the right explainer from SHAP
        explainer = shap_dict[model_name][0]
        # Calculate the shap values
        shap_values = shap_dict[model_name][1]
        # Handle regression and classification differently
        if config_dict["problem_type"] == "classification":
            # Try to get the class names
            try:
                class_names = model.classes_.tolist()
            except AttributeError:
                print("Unable to get class names automatically - classes will be encoded")
                class_names = None

            # Use SHAP's summary plot
            shap.summary_plot(
                shap_values,
                df_test,
                plot_type='violin',
                show=False,
                class_names=class_names
            )
        elif config_dict["problem_type"] == "regression":
            shap.summary_plot(
                shap_values,
                df_test,
                plot_type='violin',
                show=False
            )
        # Get the figure object
        fig = plt.gcf()
        if save:
            fname = f"{experiment_folder / 'graphs' / 'shap_summary'}_{model_name}_{'holdout'}"
            save_fig(fig, fname)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()

        # Clear keras and TF sessions/graphs etc.
        utils.tidy_tf()


def conf_matrix_plot(experiment_folder, config_dict, x_test, y_test, normalize=False, save=True):
    '''
    Creates a confusion matrix for each model. Saves them in separate files.
    '''
    # Loop over the defined models
    for model_name in config_dict["model_list"]:
        # Define the figure object
        fig, ax = plt.subplots()
        # Load the model
        try:
            model_path = glob.glob(f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}")[0]
        except IndexError:
            print("The trained model " + str('*' + model_name + '*.pkl') + " is not present")
            exit()

        print(f"Plotting Confusion Matrix for {model_name}")
        model = utils.load_model(model_name, model_path)
        # Get the predictions
        y_pred = model.predict(x_test)
        # Calc the confusion matrix
        print(y_pred)
        print(y_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        # Normalize the confusion matrix
        if normalize:
            conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        # Plot the confusion matrix
        im = ax.imshow(conf_matrix, interpolation='nearest', cmap=cmx.binary)
        ax.figure.colorbar(im, ax=ax)
        # Try to get the class names
        try:
            class_names = model.classes_.tolist()
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
            # plt.setp(
            #     ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        except AttributeError:
            print("Unable to get class names automatically")
            class_names = None
        # Setup the labels/ticks
        ax.set_xticks(np.arange(conf_matrix.shape[1]))
        ax.set_yticks(np.arange(conf_matrix.shape[0]))
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        # Add the text annotations
        fmt = '.2f' if normalize else 'd'
        # Threshold for black or white text
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                # Use white text if the colour is too dark
                ax.text(j, i, format(conf_matrix[i, j], fmt),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        # Save or show the plot
        if save:
            fname = f"{experiment_folder / 'graphs' / 'conf_matrix'}_{model_name}_{'holdout'}"
            save_fig(fig, fname)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()
        # Clear keras and TF sessions/graphs etc.
        utils.tidy_tf()


def correlation_plot(experiment_folder, config_dict, x_test, y_test, class_name, fit_line=True, save=True):
    '''
    Creates a correlation plot with a 1D line of best fit.
    '''
    # Loop over the defined models
    for model_name in config_dict["model_list"]:
        # Define the figure object
        fig, ax = plt.subplots()

        try:
            model_path = glob.glob(f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}")[0]
        except IndexError:
            print("The trained model " + str('*' + model_name + '*.pkl') + " is not present")
            exit()

        print(f"Plotting Correlation Plot for {model_name}")
        model = utils.load_model(model_name, model_path)
        # Get the predictions
        y_pred = model.predict(x_test)
        # Calc the confusion matrix
        ax.scatter(y_test, y_pred, c='black', s=5, alpha=0.8)
        # Set the axis labels
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        # Set the title
        ax.set_title(f"Correlation for {class_name} using {model_name}")
        # Add a best fit line
        if fit_line:
            ax.plot(
                np.unique(y_test),
                np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)),
                linestyle='dashed',
                linewidth=2,
                color='dimgrey'
            )
        # Save or show the plot
        if save:
            fname = f"{experiment_folder / 'graphs' / 'corr_scatter'}_{model_name}_{'holdout'}"
            save_fig(fig, fname)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()
        # Clear keras and TF sessions/graphs etc.
        utils.tidy_tf()


def histograms(experiment_folder, config_dict, x_test, y_test, class_name, save=True):
    '''
    Shows the histogram distribution of the true and predicted labels/values.

    Provides two figures side-by-side for better illustration.
    '''
    # Loop over the defined models
    for model_name in config_dict["model_list"]:

        # Load the model
        try:
            model_path = glob.glob(f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}")[0]
        except IndexError:
            print("The trained model " + str('*' + model_name + '*.pkl') + " is not present")
            exit()

        print(f"Plotting histogram for {model_name}")
        model = utils.load_model(model_name, model_path)
        # Get the predictions
        y_pred = model.predict(x_test)

        plt.hist([y_test, y_pred], label=['True', 'Predicted'], alpha=0.5, bins=50)
        plt.legend(loc='upper right')
        axes = plt.gca()
        axes.set_ylim([0, 27])
        fig = plt.gcf()

        if save:
            fname = f"{experiment_folder / 'graphs' / 'hist_overlap'}_{model_name}_{'holdout'}"
            save_fig(fig, fname)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()

        # Clear keras and TF sessions/graphs etc.
        utils.tidy_tf()


def distribution_hist(experiment_folder, config_dict, x_test, y_test, class_name, save=True):
    '''
    Shows the histogram distribution of the true and predicted labels/values.

    Provides two figures side-by-side for better illustration.
    '''
    # Loop over the defined models
    for model_name in config_dict["model_list"]:
        # Define the figure object
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 10))

        # Load the model
        try:
            model_path = glob.glob(f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}")[0]
        except IndexError:
            print("The trained model " + str('*' + model_name + '*.pkl') + " is not present")
            exit()

        print(f"Plotting histogram for {model_name}")
        model = utils.load_model(model_name, model_path)
        # Get the predictions
        y_pred = model.predict(x_test)
        # Left histograms
        ax_left.hist(y_test, bins=20, zorder=1, color='black', label="True")
        ax_left.hist(y_pred, bins=20, zorder=2, color='grey', label="Predicted")
        # Right histograms (exactly the same, just different zorder)
        ax_right.hist(y_test, bins=20, zorder=2, color='black', label="True")
        ax_right.hist(y_pred, bins=20, zorder=1, color='grey', label="Predicted")
        # Set the title
        # fig.suptitle(f"Histogram for {class_name} using {pretty_names(model_name, 'model')}", fontsize='xx-large')
        # Create a single legend
        handles, labels = ax_right.get_legend_handles_labels()
        # Add the legend
        fig.legend(handles, labels, loc='right')
        if save:
            fname = f"{experiment_folder / 'graphs' / 'hist'}_{model_name}_{'holdout'}"
            save_fig(fig, fname)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()
        # Clear keras and TF sessions/graphs etc.
        utils.tidy_tf()


def joint_plot(experiment_folder, config_dict, x_test, y_test, class_name, kind="reg", save=True):
    '''
    Uses seaborn's jointplot to illustrate correlation and distribution.
    '''
    # Loop over the defined models
    for model_name in config_dict["model_list"]:
        # Load the model
        try:
            model_path = glob.glob(f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}")[0]
        except IndexError:
            print("The trained model " + str('*' + model_name + '*.pkl') + " is not present")
            exit()

        print(f"Plotting joint plot for {model_name}")
        model = utils.load_model(model_name, model_path)
        # Get the predictions
        y_pred = model.predict(x_test)
        sns.set(style="white")
        # Make the joint plot
        plot = sns.jointplot(
            x=y_test, y=y_pred, color='grey', kind=kind,
            # Space between the marginal and main axes
            space=0,
            # Limit on the axis
            # xlim=(15, 75), ylim=(15, 75),
            # if(kind='kde'):
            #    shade_lowest=False, #gives error for regular joint

            # Control scatter properties
            scatter_kws={'s': 4},
            # Control line properties
            line_kws={'linewidth': 1.5, 'color': 'black', 'linestyle': 'dashed'},
            # Control histogram properties
            marginal_kws={'color': 'midnightblue'}
        )

        # Set the labels
        plot.ax_joint.set_xlabel('True Value')
        plot.ax_joint.set_ylabel('Predicted Value')
        x0, x1 = plot.ax_joint.get_xlim()
        y0, y1 = plot.ax_joint.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        plot.ax_joint.plot(lims, lims, ':k')

        # Extract only the pearson val (ignore the p-value)
        pearson = lambda x, y: sp.pearsonr(x, y)[0]
        # Add this value to the graph (warning: deprecated in the future)
        plot.annotate(pearson, loc='upper right', stat="Pearson's", borderpad=0.2, mode=None, edgecolor=None)
        if save:
            if kind == "kde":
                fname = f"{experiment_folder / 'graphs' / 'joint_kde'}_{model_name}_{'holdout'}"
            else:
                fname = f"{experiment_folder / 'graphs' / 'joint'}_{model_name}_{'holdout'}"
            save_fig(plot, fname)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()
        # Clear keras and TF sessions/graphs etc.
        utils.tidy_tf()


def permut_importance(experiment_folder, config_dict, scorer_dict, feature_names, data, labels, num_features, cv=None,
                      save=True):
    '''
    Use ELI5's permutation importance to assess the importance of the features.

    Note that in scikit-learn 0.21 there should be a version of this in the new model inspection module.
    This may be useful to use/watch for the future.
    '''

    print(feature_names)
    print(type(feature_names))

    # Loop over the defined models
    for model_name in config_dict["model_list"]:
        if model_name == "mlp_ens":
            continue
        # Define the figure object
        fig, ax = plt.subplots()
        # Load the model
        try:
            model_path = glob.glob(f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}")[0]
        except IndexError:
            print("The trained model " + str('*' + model_name + '*.pkl') + " is not present")
            exit()
        print(f"Plotting permutation importance for {model_name}")

        print("Model path")
        print(model_path)
        print("Model name")
        print(model_name)

        model = utils.load_model(model_name, model_path)
        # Select the scoring function
        scorer_func = scorer_dict[config_dict['fit_scorer']]
        # Handle the custom model
        if isinstance(model, tuple(CustomModel.__subclasses__())):
            # Remove the test data to avoid any saving
            if model.data_test is not None:
                model.data_test = None
            if model.labels_test is not None:
                model.labels_test = None
        # Handle the CustomModel to avoid resaving
        if model_name in CustomModel.custom_aliases:
            importances = eli5.sklearn.PermutationImportance(
                model,
                scoring=scorer_func,
                random_state=config_dict["seed_num"],
                cv=cv
            ).fit(data, labels, save_best=False)
        else:
            importances = eli5.sklearn.PermutationImportance(
                model,
                scoring=scorer_func,
                random_state=config_dict["seed_num"],
                cv=cv
            ).fit(data, labels)

        a = np.asarray(importances.results_)

        # Get the top x indices of the features
        top_indices = np.argsort(np.median(a, axis=0))[::-1][:num_features]

        # Get the names of these features
        top_features = feature_names.values[top_indices]

        # Get the top values
        top_values = a[:, top_indices]
        # Split the array up for the boxplot func
        top_values = [top_values[:, i] for i in range(top_values.shape[1])]

        top_feature_info = {'Features_names': top_features,
                            'Features_importance_value': top_values}

        df_topfeature_info = pd.DataFrame(top_feature_info, columns=["Features_names", "Features_importance_value"])

        if (cv == 'prefit'):
            df_topfeature_info.to_csv(
                f"{experiment_folder / 'results' / 'permutimp_TopFeatures_info'}_{model_name}" + ".csv")
        else:
            df_topfeature_info.to_csv(
                f"{experiment_folder / 'results' / 'permutimp_TopFeatures_info'}_{model_name}_cv-{cv}" + ".csv")

        # Make a horizontal boxplot ordered by the magnitude
        ax = sns.boxplot(
            x=top_values,
            y=top_features,
            orient="h",
            ax=ax
        )
        if config_dict["problem_type"] == "classification":
            ax.set_xlabel(f"{pretty_names(config_dict['fit_scorer'], 'score')} Decrease")
        else:
            ax.set_xlabel(f"{pretty_names(config_dict['fit_scorer'], 'score')} Increase")
            ax.set_ylabel("Features")

        # Do a np.any(<0) check to see if we get negative values
        # These indicate that shuffling the feature actually improves performance
        # Save the plot
        if save:
            fname = f"{experiment_folder / 'graphs' / 'permutimp'}_{model_name}_{'holdout'}"
            save_fig(fig, fname)

        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()
        # Clear keras and TF sessions/graphs etc.
        utils.tidy_tf()


def shap_plots(experiment_folder, config_dict, feature_names, x, x_test, y_test, x_train, num_top_features,
               pcAgreementLevel=10, save=True):
    if (config_dict["explanations_data"] == "all" or "test" or "train" or "exemplars"):
        data_forexplanations = config_dict["explanations_data"]
    # assume test set
    else:
        data_forexplanations = "train"

    if (len(feature_names) <= num_top_features):
        num_top = len(feature_names)
    else:
        num_top = num_top_features

    # Convert the data into dataframes to ensure features are displayed
    df_train = pd.DataFrame(data=x_train, columns=feature_names)
    print(feature_names)
    print(len(feature_names))

    # Loop over the defined models
    for model_name in config_dict["model_list"]:

        # Load the model
        try:
            model_path = glob.glob(f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}")[0]
        except IndexError:
            print("The trained model " + str('*' + model_name + '*.pkl') + " is not present")
            exit()

        print("Model path")
        print(model_path)
        print("Model name")
        print(model_name)

        print(f"Plotting SHAP plots for {model_name}")

        model = utils.load_model(model_name, model_path)

        # Select the right explainer from SHAP
        explainer = utils.select_explainer(model, model_name, df_train, config_dict["problem_type"])

        # Get the exemplars on the test set -- maybe to modify to include probability
        exemplar_X_test = utils.get_exemplars(x_test, y_test, model, config_dict, pcAgreementLevel)

        # Compute SHAP values for desired data (either test set x_test, or exemplar_X_test or the entire dataset x)
        if (data_forexplanations == "all"):
            shap_values = explainer.shap_values(x)
            data = x
        elif (data_forexplanations == "train"):
            shap_values = explainer.shap_values(x_train)
            data = x_train
        elif (data_forexplanations == "test"):
            shap_values = explainer.shap_values(x_test)
            data = x_test
        elif (data_forexplanations == "exemplars"):
            shap_values = explainer.shap_values(exemplar_X_test)
            data = exemplar_X_test
        # otherwise assume train set
        else:
            shap_values = explainer.shap_values(x_train)
            data = x_train

        # Handle regression and classification differently and store the shap_values in shap_values_selected

        # Classification
        if config_dict["problem_type"] == "classification":

            # For classification there is not difference between data structure returned by SHAP
            shap_values_selected = shap_values

            # Try to get the class names
            try:
                class_names = model.classes_.tolist()
            except AttributeError:
                print("Unable to get class names automatically - classes will be encoded")
                class_names = None

            # Produce and save SHAP bar plot

            if (model_name == 'xgboost' and len(class_names) == 2):
                # Use SHAP's summary plot
                shap.summary_plot(
                    shap_values_selected,
                    data,
                    plot_type='bar',
                    max_display=num_top,
                    color=plt.get_cmap("Set3"),
                    feature_names=feature_names,
                    show=False,
                    class_names=class_names
                )
                fig = plt.gcf()

                # Save the plot for multi-class classification
                if save:
                    fname = f"{experiment_folder / 'graphs' / 'shap_bar_plot'}_{data_forexplanations}_{model_name}_{'holdout'}"
                    save_fig(fig, fname)
                plt.draw()
                plt.tight_layout()
                plt.pause(0.001)
                time.sleep(2)
                # Close the figure to ensure we start anew
                plt.clf()
                plt.close()

            else:
                # Use SHAP's summary plot
                shap.summary_plot(
                    shap_values_selected,
                    data,
                    plot_type='bar',
                    max_display=num_top,
                    feature_names=feature_names,
                    show=False,
                    class_names=class_names
                )
                fig = plt.gcf()

                # Save the plot for multi-class classification
                if save:
                    fname = f"{experiment_folder / 'graphs' / 'shap_bar_plot'}_{data_forexplanations}_{model_name}_{'holdout'}"
                    save_fig(fig, fname)
                plt.draw()
                plt.tight_layout()
                plt.pause(0.001)
                time.sleep(2)
                # Close the figure to ensure we start anew
                plt.clf()
                plt.close()

            objects, abundance, shap_values_mean_sorted = utils.compute_average_abundance_top_features(config_dict,
                                                                                                       num_top,
                                                                                                       model_name,
                                                                                                       class_names,
                                                                                                       feature_names,
                                                                                                       data,
                                                                                                       shap_values_selected)

            summary_SHAPdotplot_perclass(experiment_folder, class_names, model_name, feature_names,
                                         num_top, data, shap_values_selected)

            plt.clf()
            plt.close()


        # Regression
        else:

            # Produce and save bar plot for regression

            # Handle Shap saves differently the values for Keras when it's regression
            if (model_name == 'mlp_keras'):
                shap_values_selected = shap_values[0]
            else:
                shap_values_selected = shap_values

            # Plot shap bar plot
            shap.summary_plot(
                shap_values_selected,
                data,
                plot_type='bar',
                color_bar='000',
                max_display=num_top,
                feature_names=feature_names,
                show=False
            )
            fig = plt.gcf()
            # fig.set_size_inches(30,30, forward=True)
            # Setup the title
            # fig.suptitle(f"SHAP Summary Plot for top features {pretty_names(model_name, 'model')} for {class_col} ({name})", fontsize=16, y=1.4)

            # Save the plot
            if save:
                fname = f"{experiment_folder / 'graphs' / 'shap_bar_plot'}_{data_forexplanations}_{model_name}_{'holdout'}"
                save_fig(fig, fname)

                # img = mp_img.imread(fname+'.png')
            # plt.imshow(img)

            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
            time.sleep(2)
            # Close the figure to ensure we start anew
            plt.clf()
            plt.close()

            #  #Produce and save dot plot for regression

            shap.summary_plot(
                shap_values_selected,
                data,
                plot_type='dot',
                color_bar='000',
                max_display=num_top,
                feature_names=feature_names,
                show=False
            )
            fig = plt.gcf()
            # Save the plot
            if save:
                fname = f"{experiment_folder / 'graphs' / 'shap_dot_plot'}_{data_forexplanations}_{model_name}_{'holdout'}"
                save_fig(fig, fname)

            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
            time.sleep(2)

            # Close the figure to ensure we start anew
            plt.clf()
            plt.close()

            # Plot abundance bar plot feature from SHAP
            class_names = []
            objects, abundance, shap_values_mean_sorted = utils.compute_average_abundance_top_features(config_dict,
                                                                                                       num_top,
                                                                                                       model_name,
                                                                                                       class_names,
                                                                                                       feature_names,
                                                                                                       data,
                                                                                                       shap_values_selected)

        # Displaying the average percentage %
        abundance = np.asarray(abundance) / 10

        d = {'Features': objects,
             'SHAP values': shap_values_mean_sorted,
             'Average abundance': list(abundance)}

        fname = f"{experiment_folder / 'results' / 'top_features_AbsMeanSHAP_Abundance'}_{data_forexplanations}_{model_name}_{'holdout'}"
        df = pd.DataFrame(d)
        df.to_csv(fname + '.csv')

        # Bar plot of average abundance across all the samples of the top genera

        y_pos = np.arange(len(objects))
        plt.barh(y_pos, abundance, align='center', color='black')
        plt.yticks(y_pos, objects)
        plt.gca().invert_yaxis()
        plt.xlabel('Average abundance (%)')

        fig = plt.gcf()

        if save:
            fname = f"{experiment_folder / 'graphs' / 'abundance_top_features_exemplars'}_{data_forexplanations}_{model_name}_{'holdout'}"
            save_fig(fig, fname)

        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()

        # Clear keras and TF sessions/graphs etc.
        utils.tidy_tf()


def shap_force_plots(experiment_folder, config_dict, x_test, y_test, feature_names, x, y, x_train, data_forexplanations,
                     class_col="?",
                     top_exemplars=0.1, save=True):
    '''
       Wrapper to create a SHAP force plot for the top exemplar of each class for each model.
       '''
    # Convert the data into dataframes to ensure features are displayed
    if (data_forexplanations == "all"):
        data = x
        y_data = y
    elif (data_forexplanations == "test"):
        data = x_test
        y_data = y_test

    # Convert the data into dataframes to ensure features are displayed
    df_data = pd.DataFrame(data=data, columns=feature_names)
    df_train = pd.DataFrame(data=x_train, columns=feature_names)

    # Get the model paths
    for model_name in config_dict["model_list"]:

        try:
            model_path = glob.glob(f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}")[0]
        except IndexError:
            print("The trained model " + str('*' + model_name + '*.pkl') + " is not present")
            exit()

        print(f"Plotting SHAP for {model_name}")
        model = utils.load_model(model_name, model_path)

        # Select the right explainer from SHAP
        explainer = utils.select_explainer(model, model_name, df_train, config_dict["problem_type"])
        shap_values = explainer.shap_values(data)

        # Handle classification and regression differently
        if config_dict["problem_type"] == "classification":
            # Try to get the class names
            try:
                class_names = model.classes_.tolist()
            except AttributeError:
                print("Unable to get class names automatically - classes will be encoded")
                # Hack to get numbers instead - should probably raise an error
                class_names = range(100)
            # Get the predicted probabilities
            probs = model.predict_proba(data)
            # Use a masked array to check which predictions are correct, and then which we're most confident in
            class_exemplars = np.ma.masked_array(
                probs,
                mask=np.repeat(model.predict(data) != y_data, probs.shape[1])
                # Need to repeat so the mask is the same shape as predict_proba
            ).argmax(0).tolist()
            # print(class_exemplars)
            for i, (class_index, class_name) in enumerate(zip(class_exemplars, class_names)):
                # Close the figure to ensure we start anew
                plt.clf()
                plt.close()
                # exemplar_data = df_test.iloc[class_index, :]
                exemplar_data = data[class_index, :]
                # Create the force plot
                fig = shap.force_plot(
                    explainer.expected_value[i],
                    shap_values[i][class_index],
                    exemplar_data,
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False,
                    text_rotation=30
                )
                # Need to add label/text on the side for the class name
                print(f"{pretty_names(model_name, 'model')}")
                fig.suptitle(
                    f"SHAP Force Plot for top exemplar using {pretty_names(model_name, 'model')} with class {class_name}",
                    fontsize=16, y=1.4)
                # Save the plot
                if save:
                    fname = f"{experiment_folder / 'graphs' / 'shap_force_single'}_{model_name}_class{class_name}_{'holdout'}"
                    save_fig(fig, fname)
                plt.draw()
                plt.tight_layout()
                plt.pause(0.001)
                time.sleep(2)
                # Close the figure to ensure we start anew
                plt.clf()
                plt.close()

        # Different exemplar calc for regression
        elif config_dict["problem_type"] == "regression":
            # Containers to avoid repetition with calling graph func
            names = []
            exemplar_indices = []
            # Get the predictions
            preds = model.predict(data).flatten()
            # Calculate the difference in predictions
            dists = np.abs(y_data - preds)
            # Select the max and min top exemplars (i.e. closest to the max and min values for the target)
            if top_exemplars is not None:
                indices = dists.argsort()
                # Select the top percentage to choose from
                num_indices = int(len(preds) * top_exemplars)
                # Select the top exemplars
                exemplar_index = indices[:num_indices]
                # With clashes, it takes the first found (which is good as this corresponds to the lower prediction error)
                top_min_index = exemplar_index[np.argmin(y_data[exemplar_index])]
                exemplar_indices.append(top_min_index)
                names.append("min")
                top_max_index = exemplar_index[np.argmax(y_data[exemplar_index])]
                exemplar_indices.append(top_max_index)
                names.append("max")
            # Otherwise we just take our single best prediction
            else:
                exemplar_indices.append(dists.argmin())
                names.append("closest")
            # Create a plot for each of the selected exemplars
            for name, exemplar_index in zip(names, exemplar_indices):
                # Create the plot
                fig = shap.force_plot(
                    explainer.expected_value,
                    shap_values[exemplar_index],
                    data[exemplar_index],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False,
                    text_rotation=30
                )
                # Setup the title
                fig.suptitle(
                    f"SHAP Force Plot for top exemplar using {pretty_names(model_name, 'model')} for {class_col} ({name})",
                    fontsize=16, y=1.4)
                # Save the plot
                if save:
                    fname = f"{experiment_folder / 'graphs' / 'shap_force_single'}_{model_name}_{name}_{'holdout'}"
                    save_fig(fig, fname)
                plt.draw()
                plt.tight_layout()
                plt.pause(0.001)
                time.sleep(2)
                # Close the figure to ensure we start anew
                plt.clf()
                plt.close()
                # Clear everything
        # Clear keras and TF sessions/graphs etc.
        utils.tidy_tf()


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

    if (config_dict["data_type"] == "microbiome"):
        # This reads and preprocesses microbiome data using calour library -- it would be better to change this preprocessing so that it is not dependent from calour
        x, y, features_names = train_models.get_data_microbiome(config_dict["file_path"], config_dict["metadata_file"],
                                                                config_dict)
        x_heldout, y_heldout, features_names = train_models.get_data_microbiome(config_dict["file_path_holdout_data"],
                                                                                config_dict[
                                                                                    "metadata_file_holdout_data"],
                                                                                config_dict)
    elif (config_dict["data_type"] == "gene_expression"):
        # This reads and preprocesses microbiome data using calour library -- it would be better to change this preprocessing so that it is not dependent from calour
        x, y, features_names = train_models.get_data_gene_expression(config_dict["file_path"],
                                                                     config_dict["metadata_file"],
                                                                     config_dict)
        x_heldout, y_heldout, features_names = train_models.get_data_gene_expression(
            config_dict["file_path_holdout_data"],
            config_dict["metadata_file_holdout_data"],
            config_dict)
    elif (config_dict["data_type"] == "metabolomic"):
        x, y, features_names = train_models.get_data_metabolomic(config_dict["file_path"], config_dict["metadata_file"],
                                                                 config_dict)
        x_heldout, y_heldout, features_names = train_models.get_data_metabolomic(config_dict["file_path_holdout_data"],
                                                                                 config_dict[
                                                                                     "metadata_file_holdout_data"],
                                                                                 config_dict)
    elif (config_dict["data_type"] == "tabular"):
        x, y, features_names = train_models.get_data_tabular(config_dict["file_path"], config_dict["metadata_file"],
                                                             config_dict)

        x_heldout, y_heldout, features_names = train_models.get_data_tabular(config_dict["file_path_holdout_data"], config_dict["metadata_file_holdout_data"], config_dict)
    else:
        # At the moment for all the other data types, for example metabolomics, we have not implemented preprocessing except for standardisation with StandardScaler()
        x, y, features_names = train_models.get_data(config_dict["file_path"], config_dict["target"],
                                                     config_dict["metadata_file"])
        x_heldout, y_heldout, features_names = train_models.get_data(config_dict["file_path_holdout_data"],
                                                                     config_dict["metadata_file_holdout_data"],
                                                                     config_dict)

    # Split the data in train and test
    if config_dict["stratify_by_groups"] == "Y":

        gss = GroupShuffleSplit(n_splits=1, test_size=config_dict["test_size"], random_state=config_dict["seed_num"])
        # gss=GroupKFold(n_splits=5)

        metadata = pd.read_csv(config_dict["metadata_file"], index_col=0, sep='\t')
        le = LabelEncoder()
        groups = le.fit_transform(metadata[config_dict["groups"]])
        for train_idx, test_idx in gss.split(x, y, groups):
            x_train, x_test, y_train, y_test = x[train_idx], x[test_idx], y[train_idx], y[test_idx]

    else:
        x_train, x_test, y_train, y_test = models.split_data(
            x, y, config_dict["test_size"], config_dict["seed_num"],
            config_dict["problem_type"]
        )
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

    # Create the folders needed
    experiment_folder = utils.create_experiment_folders(config_dict, config_path)
    # Select only the scorers that we want
    scorer_dict = models.define_scorers(config_dict["problem_type"])
    scorer_dict = {k: scorer_dict[k] for k in config_dict["scorer_list"]}
    # See what plots are defined
    plot_dict = define_plots(config_dict["problem_type"])

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

        if model_name in CustomModel.custom_aliases:
            # Pickling doesn't inherit the self.__class__.__dict__, just self.__dict__
            # So set that up here
            # Other option is to modify cls.__getstate__
            CustomModel.custom_aliases[model_name].setup_cls_vars(config_dict, experiment_folder)

        # Load the model
        try:
            model_path = glob.glob(f"{experiment_folder / 'models' / str('*' + model_name + '*.pkl')}")[0]
        except IndexError:
            print("The trained model " + str('*' + model_name + '*.pkl') + " is not present")
            exit()

        print(f"Plotting barplot for {model_name} using {config_dict['fit_scorer']}")
        model = utils.load_model(model_name, model_path)


        # Evaluate the best model using all the scores and CV
        performance_results_dict = models.evaluate_model(model, config_dict['problem_type'], x_train, y_train,
                                                  x_heldout, y_heldout)
        # Save the results
        df_performance_results, fname_perfResults = models.save_results(
            results_folder, df_performance_results, performance_results_dict,
            model_name, fname,
            suffix="_performance_results_holdout", save_pkl=False, save_csv=True)

        print(f"{model_name} evaluation on hold out complete! Results saved at {Path(fname_perfResults).parents[0]}")



    # Central func to define the args for the plots
    plot_graphs(config_dict, experiment_folder, features_names, plot_dict, x, y, x_train, y_train, x_heldout, y_heldout,
                scorer_dict)