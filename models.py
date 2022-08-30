##### Fix for each thread spawning its own GUI is to use 1 thread 
##### Change this to n_jobs = -1 for all-core processing (when we get that working)
n_jobs = -1

import json
import pickle
from json.decoder import JSONDecodeError
from pathlib import Path
import pdb
import warnings

import numpy as np
import scipy.sparse
import pandas as pd
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV#, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer, roc_auc_score
import sklearn.metrics as skm
from sklearn.preprocessing import normalize

import joblib
from xgboost import XGBClassifier, XGBRegressor

import model_params

from custom_model import CustomModel
from custom_model import FixedKeras, AutoKeras, AutoSKLearn, AutoLGBM, AutoXGBoost #, AutoGluon

import logging
omicLogger = logging.getLogger("OmicLogger")

from utils import copy_best_content
import os

from plotting import plot_model_performance


########## LOAD/DEFINE ##########
def load_params_json(fpath):
    '''
    Function to load and process parameters for random search defined in a JSON

    N.B.: This has since been replaced by defining them in a separate Python script
    '''
    omicLogger.debug('Load parameters from json (Redundant?)...')
    # First load the JSON file in as a dict
    print(fpath)
    try:
        with open(fpath) as json_file:
            param_dict = json.load(json_file)
    except JSONDecodeError:
        print(f"File {fpath} not found or not a valid JSON file")
        raise
    # Separate container to avoid issues
    param_ranges = {}
    # Loop over the items in the JSON file
    for key, value in param_dict.items():
        if key == "extras":
            continue
        # If it's a list, create the range of values
        if isinstance(value, list):
            # Create a range if specified
            if value[0] == "range":
                value = value[1:]
                # Get a range of values (but cast to int if they're ints)
                if isinstance(value[0], int):
                    start, stop, num_vals = value
                    param_ranges[key] = [int(i) for i in np.linspace(start, stop, num=num_vals)]
                # Same as above but using gloats
                elif isinstance(value[0], float):
                    start, stop, num_vals = value
                    param_ranges[key] = [float(i) for i in np.linspace(start, stop, num=num_vals)]
            # Nothing needs to be done with a list of strings
            elif isinstance(value[0], (str, int, float)):
                param_ranges[key] = value
            else:
                raise TypeError(f"{value} of type {type(value)} is not a valid parameter in a list")
        # Single values can be assigned
        elif isinstance(value, (int, float, str)):
            # May need to cast this to a list just so we can add args (the 'extras') with ease
            # Should not effect RandomizedSearchCV
            param_ranges[key] = value
        else:
            raise TypeError(f"{value} of type {type(value)} is not a valid parameter")
    # Extras is useful for when you mix 'None' values with numerical ranges
    if "extras" in param_dict:
        extras = param_dict["extras"]
        # If there are extra arguments then add them here
        for key, values in extras.items():
            try:
                print(key, param_ranges)
                param_ranges[key].extend(values)
            except KeyError:
                print(f"{key} does not exist in {param_ranges}")
                raise
            except AttributeError:
                print(f"Cannot add extra args to {key} with value {values}")
                raise
    return param_ranges

def define_scorers(problem_type):
    '''
    Define the different measures we can use
    '''
    omicLogger.debug('Define scores to evaluate the models...')
    
    if problem_type == "classification":
        scorer_dict = {
            'acc': skm.make_scorer(skm.accuracy_score),
            'f1': skm.make_scorer(skm.f1_score, average='weighted'),
            # 'f1_class': accuracy_score(true_labels, pred_labels), #just added
            'prec': skm.make_scorer(skm.precision_score, average='weighted'),
            # 'prec_class': skm.make_scorer(skm.precision_score, average=None),
            'recall': skm.make_scorer(skm.recall_score, average='weighted')
            # 'recall_class': skm.make_scorer(skm.recall_score, average=None)
            # 'cv_f1': skm.make_scorer(cross_val_score, scoring='f1_weighted', cv=5)
        }
    elif problem_type == "regression":
        scorer_dict = {
            'mse': skm.make_scorer(skm.mean_squared_error, greater_is_better=False),
            'mean_ae': skm.make_scorer(skm.mean_absolute_error, greater_is_better=False),
            'med_ae': skm.make_scorer(skm.median_absolute_error, greater_is_better=False),
            'rmse': skm.make_scorer(rmse, greater_is_better=False),
            'mean_ape' : skm.make_scorer(skm.mean_absolute_percentage_error, greater_is_better=False),
            'r2' : skm.make_scorer(skm.r2_score, greater_is_better=True)
        }
    else:
        raise ValueError(f"{problem_type} is not recognised, must be either 'regression' or 'classification'")
    return scorer_dict

def select_model_dict(hyper_tuning):
    '''
    Select what parameter range specific we are using based on the given hyper_tuning method.
    '''
    omicLogger.debug('Get tunning settings...')
    
    if hyper_tuning == "random":
        ref_model_dict = model_params.sk_random
    elif hyper_tuning == "grid":
        ref_model_dict = model_params.sk_grid
    elif hyper_tuning is None:
        ref_model_dict = model_params.single_model
    else:
        raise ValueError(f"{hyper_tuning} is not a valid option")
    return ref_model_dict

def define_models(problem_type, hyper_tuning):
    '''
    Define the models to be run.

    The name is the key, the value is a tuple with the model function, and defined params
    '''
    omicLogger.debug('Defining the set of models...')
    ref_model_dict = select_model_dict(hyper_tuning)

    if problem_type == "classification":
        try:
            # Specific modifications for problem type
            if hyper_tuning is None or hyper_tuning=="boaas":
                ref_model_dict['svm']['probability'] = True
            else:
                ref_model_dict['svm']['probability'] = [True]
        # Otherwise pass - models may not always be defined for every tuning method
        except KeyError:
            pass
        # Define dict
        model_dict = {
            'rf': (RandomForestClassifier, ref_model_dict['rf']),
            'svm': (SVC, ref_model_dict['svm']),
            'knn': (KNeighborsClassifier, ref_model_dict['knn']),
            'adaboost': (AdaBoostClassifier, ref_model_dict['adaboost']),
            'xgboost': (XGBClassifier, ref_model_dict['xgboost'])
        }
    elif problem_type == "regression":
        # Specific modifications for problem type
        
        # Define dict
        model_dict = {
            'rf': (RandomForestRegressor, ref_model_dict['rf']),
            'svm': (SVR, ref_model_dict['svm']),
            'knn': (KNeighborsRegressor, ref_model_dict['knn']),
            'adaboost': (AdaBoostRegressor, ref_model_dict['adaboost']),
            'xgboost': (XGBRegressor, ref_model_dict['xgboost'])
        }
    else:
        raise ValueError(f"{problem_type} is not recognised, must be either 'regression' or 'classification'")
    # The CustomModels handle classification and regression themselves so put outside
    # For mixing tuning types, default to using the single model for mlp_ens
    try:
        model_dict["fixedkeras"] = (FixedKeras, ref_model_dict['fixedkeras'])
        model_dict["autokeras"] = (AutoKeras, ref_model_dict['autokeras'])
        model_dict["autolgbm"] = (AutoLGBM, ref_model_dict['autolgbm'])
        model_dict["autoxgboost"] = (AutoXGBoost, ref_model_dict['autoxgboost'])
        model_dict["autosklearn"] = (AutoSKLearn, ref_model_dict['autosklearn'])
        # model_dict["autogluon"] = (AutoGluon, ref_model_dict['autogluon'])
    except KeyError:
        model_dict["fixedkeras"] = (FixedKeras, model_params.single_model['fixedkeras'])
        model_dict["autokeras"] = (AutoKeras, model_params.single_model['autokeras'])
        model_dict["autolgbm"] = (AutoLGBM, model_params.single_model['autolgbm'])
        model_dict["autoxgboost"] = (AutoXGBoost, model_params.single_model['autoxgboost'])
        model_dict["autosklearn"] = (AutoSKLearn, model_params.single_model['autosklearn'])
        # model_dict["autogluon"] = (AutoGluon, model_params.single_model['autogluon'])
    return model_dict


########## SAVE ##########
def save_results(results_folder, df, score_dict, model_name, fname, suffix=None, save_pkl=False, save_csv=True):
    '''
    Store the results of the latest model and save this to csv
    '''
    omicLogger.debug('Save results to file...')
    
    df = df.append(pd.Series(score_dict, name=model_name))
    fname = str(results_folder / fname)
    # Add a suffix to the filename if provided
    if suffix is not None:
        fname += suffix
    # Save as a csv
    if save_csv:
        df.to_csv(fname+".csv", index_label="model")
    # Pickle using pandas internal access to it
    if save_pkl:
        df.to_pickle(fname+".pkl")
    return df, fname

def save_model(experiment_folder, model, model_name):
    '''
    Save a given model to the model folder
    '''
    omicLogger.debug('Saving model...')
    model_folder = experiment_folder / "models"
    # THe CustomModels handle themselves
    if model_name not in CustomModel.custom_aliases:
        print(f"Saving {model_name} model")
        save_name = model_folder / f"{model_name}_best.pkl"
        with open(save_name, 'wb') as f:
            joblib.dump(model, f)
    else:  # hat: added this 
        # print(f"Saving {model_name} model")
        # save_name = model_folder / f"{model_name}_best.pkl"
        model.save_model()


########## EVALUATE ##########
def evaluate_model(model, problem_type, x_train, y_train, x_test, y_test):
    '''
    Define the different measures we can use and Calculate some of them on the model
    '''
    omicLogger.debug('Evaluate the model...')
    
    pred_test = model.predict(x_test)
    pred_train = model.predict(x_train)
    pred_out = np.concatenate((pred_train,pred_test))

    if problem_type == "classification":
        col_names = ['Prediction']
        if len(set(y_train))==2:
            pred_test_proba = model.predict_proba(x_test)[:,1]
            pred_train_proba = model.predict_proba(x_train)[:,1]
            prob_out = np.concatenate((pred_train_proba,pred_test_proba)).reshape(-1,1)
            col_names +=['probability']
        else:
            pred_test_proba = normalize(model.predict_proba(x_test), axis=1, norm='l1')
            pred_train_proba = normalize(model.predict_proba(x_train), axis=1, norm='l1')
            prob_out = np.concatenate((pred_train_proba,pred_test_proba))
            col_names += [f'class_{i}_prob' for i in range(len(set(y_train)))]
            
        score_dict = {
            'Accuracy_Train': accuracy_score(y_train, pred_train),
            'Accuracy_Test': accuracy_score(y_test, pred_test),
            'F1_score_Train': f1_score(y_train, pred_train, average='weighted'),
            'F1_score_Test': f1_score(y_test, pred_test, average='weighted'),
            'F1_score_PerClass_Train': f1_score(y_train, pred_train, average=None),
            'F1_score_PerClass_Test': f1_score(y_test, pred_test, average=None),
            'Precision_Train': precision_score(y_train, pred_train, average='weighted'),
            'Precision_Test': precision_score(y_test, pred_test, average='weighted'),
            'Precision_PerClass_Train': precision_score(y_train, pred_train, average=None),
            'Precision_PerClass_Test': precision_score(y_test, pred_test, average=None),
            'Recall_Train': recall_score(y_train, pred_train, average='weighted'),
            'Recall_Test': recall_score(y_test, pred_test, average='weighted'),
            'Recall_PerClass_Train': recall_score(y_train, pred_train, average=None),
            'Recall_PerClass_Test': recall_score(y_test, pred_test, average=None),
            'Conf_matrix_Train': confusion_matrix(y_train, pred_train),
            'Conf_matrix_Test': confusion_matrix(y_test, pred_test),
            'ROC_auc_score_Train':roc_auc_score(y_train, pred_train_proba,multi_class='ovo'),
            'ROC_auc_score_Test':roc_auc_score(y_test, pred_test_proba,multi_class='ovo'),
            # 'CV_F1Scores': cross_val_score(model, x_train, y_train, scoring='f1_weighted', cv=5)
        }
        
        pred_out = pd.DataFrame(np.concatenate((pred_out.reshape(-1,1),prob_out),axis=1),columns=col_names)
    else:
        score_dict = {
            'MSE_Train': skm.mean_squared_error(y_train, pred_train),
            'MSE_Test': skm.mean_squared_error(y_test, pred_test),
            'Mean_AE_Train': skm.mean_absolute_error(y_train, pred_train),
            'Mean_AE_Test': skm.mean_absolute_error(y_test, pred_test),
            'Med_ae_Train': skm.median_absolute_error(y_train, pred_train),
            'Med_ae_Test': skm.median_absolute_error(y_test, pred_test),
            'Mean_APE_Train': skm.mean_absolute_percentage_error(y_train, pred_train),
            'Mean_APE_Test': skm.mean_absolute_percentage_error(y_test, pred_test),
            'R2_Train': skm.r2_score(y_train, pred_train),
            'R2_Test': skm.r2_score(y_test, pred_test)
            # 'CV_F1Scores': cross_val_score(model, x_train, y_train, scoring='mean_ae', cv=5)
        }
        
        pred_out = pd.DataFrame(pred_out,columns=['Prediction'])

    return score_dict, pred_out

def eval_scores(problem_type, scorer_dict, model, data, true_labels):
    omicLogger.debug('Gathering evaluation scores...')
    scores_dict = {}
    for score_name, score_func in scorer_dict.items():
        if problem_type == "regression":
            scores_dict[score_name] = np.abs(score_func(model, data, true_labels))
        else:
            scores_dict[score_name] = score_func(model, data, true_labels)

    return scores_dict

def rmse(y_true, y_pred):
    omicLogger.debug('calculating rmse...')
    return np.sqrt(skm.mean_squared_error(y_true, y_pred))

def best_selector(experiment_folder,problem_type,metric=None,collapse_tax=None):
    """
    Give trained models this will find and select the best one
    """
        
    if collapse_tax == None:
        collapse_tax = ''
    
    omicLogger.debug("selecting best model...")
    filepath = experiment_folder/f'results/scores_{collapse_tax}_performance_results_testset.csv'
    
    if not os.path.exists(filepath):
        raise ValueError(f'{filepath} does not exist')
    
    df = pd.read_csv(filepath)
    df = df.set_index('model')
    
    if problem_type=="classification":
        if (metric == None):
            omicLogger.info(f'Best selection metric is None, Defaulting to F1_score...')
            metric = 'f1'
        low=False
    else:
        if (metric == None):
            omicLogger.info(f'Best selection metric is None, Defaulting to Mean_AE...')
            metric = 'mean_ae'
        low=True
        
    df_cols = list(set([x.replace('_Train','').replace('_Test','') for x in list(df.columns) if ('PerClass' not in x)]))
    offical_name = [x for x in df_cols if (metric in x.lower())]
        
    if len(offical_name)==0:
        raise ValueError(f'{metric} not in metrics calculated for models')
    
    metric = offical_name[0]
    
    t_df = df[[metric+'_Train',metric+'_Test']]
    
    plot_model_performance(experiment_folder,t_df,metric,low=low)
    
    ang = t_df.apply(lambda row : round(np.arccos(np.dot(row.values,[1,1])/(np.linalg.norm(row.values)*np.linalg.norm([1,1]))),4),axis=1)
    ang.name='Angle'

    nrm = t_df.apply(lambda row : round(np.linalg.norm(row.values-1+int(low)),4),axis=1)
    nrm.name='Norm'
    
    best = pd.concat([nrm,ang],axis=1)
    best.sort_values(by=['Norm','Angle'],inplace=True)

    nrm_min = list(np.where(best['Norm']==best['Norm'].min())[0])
    sub1 = best['Angle'].iloc[nrm_min]
    ang_min = list(np.where(sub1==sub1.min())[0])
    best_models = list(sub1.keys()[ang_min])
    
    return best_models

########## WRAPPERS ##########
def random_search(model, model_name, param_ranges, budget, x_train, y_train, seed_num, scorer_dict, fit_scorer):
    '''
    Wrapper for using sklearn's RandomizedSearchCV
    '''
    omicLogger.debug('Training with a random search...')
    # If possible, set the random state for the model
    try:
        # Just a dummy to see if the model has a random state attribute
        # Improvement would be if there is hasattr func but for arguments
        _ = model(random_state=0)
        param_ranges["random_state"] = [seed_num]
    except TypeError:
        pass
    # Setup the random search with cross val
    print("Setup the random search with cross val")
    random_search = RandomizedSearchCV(
        estimator=model(),
        param_distributions=param_ranges,
        n_iter=budget,
        cv=5,
        verbose=1,
        n_jobs=n_jobs,
        random_state=seed_num,
        pre_dispatch="2*n_jobs",
        scoring=scorer_dict,
        refit=fit_scorer
    )

    # Fit the random search
    print("Fit the random search")
    try:
        random_search.fit(x_train, y_train)
    except ValueError:
        print("!!! ERROR - PLEASE SELECT VALID TARGET AND PREDICTION TASK")
        raise 
    # Return the best estimator found
    print(random_search.best_estimator_)
    return random_search.best_estimator_

def grid_search(model, model_name, param_ranges, x_train, y_train, seed_num, scorer_dict, fit_scorer):
    '''
    Wrapper for using sklearn's GridSearchCV
    '''
    omicLogger.debug('Training with a grid search...')
    try:
        _ = model(random_state=0)
        param_ranges["random_state"] = [seed_num]
    except TypeError:
        pass

    grid_search = GridSearchCV(
        estimator=model(),
        param_grid=param_ranges,
        cv=5,
        verbose=1,
        n_jobs=n_jobs,
        pre_dispatch="2*n_jobs",
        scoring=scorer_dict,
        refit=fit_scorer
    )
    # Fit the random search
    grid_search.fit(x_train, y_train)
    # Return the best estimator found
    print(grid_search.best_estimator_)
    return grid_search.best_estimator_

def single_model(model, param_ranges, x_train, y_train, seed_num):
    '''
    Wrapper for training and setting up a single model (i.e. no tuning).
    '''
    omicLogger.debug('Training as single model...')
    try:
        _ = model(random_state=0)
        param_ranges["random_state"] = seed_num
    except TypeError:
        pass
    print(model().set_params(**param_ranges))
    trained_model = model().set_params(**param_ranges).fit(x_train, y_train)
    return trained_model


########## RUN MODELS ##########
def predict_model(model, x_train, y_train, x_test=None):
    '''
    Generic function to fit a model and return predictions on train and test data (if given)
    '''
    omicLogger.debug('Predicting with given model...')
    model.fit(x_train, y_train)
    train_preds = model.predict(x_train)
    if x_test is not None:
        test_preds = model.predict(x_test)
    else:
        test_preds = None
    return train_preds, test_preds

def run_models(
    config_dict, model_list, model_dict, df_train, df_test, x_train, y_train, x_test, 
    y_test, collapse_tax, experiment_folder, remove_class, merge_class, scorer_dict,
    fit_scorer, hyper_tuning, hyper_budget,  problem_type, seed_num):
    '''
    Run (and tune if applicable) each of the models sequentially, saving the results and models.
    '''
    omicLogger.debug('Initialised training & tuning of models...')
    
    # Construct the filepath to save the results
    results_folder = experiment_folder / "results"

    # Create dataframe for performance results
    df_performance_results = pd.DataFrame()


    if(config_dict["data_type"]=="microbiome"):
        #This is specific to microbiome
        fname = f"scores_{collapse_tax}"
    else:
        fname = "scores_"

    #Remove or merge samples based on target values (for example merging to categories, if classification)
    if remove_class is not None:
        fname += "_remove"
    elif merge_class is not None:
        fname += "_merge"

    # Just need it here for determing tuning logic
    ref_model_dict = select_model_dict(hyper_tuning)
    # So that we can pass the func to the CustomModels
    scorer_func = scorer_dict[config_dict['fit_scorer']]

    # Run each model
    for model_name in model_list:
        omicLogger.debug(f'Training model: {model_name}')
        print(f"Testing {model_name}")
        # Placeholder variable to handle mixed hyperparam tuning logic for MLPEnsemble
        single_model_flag = False
        # Load the model and it's parameter path
        model, param_ranges = model_dict[model_name]
        # Setup the CustomModels
        if model_name in CustomModel.custom_aliases:
            single_model_flag, param_ranges = model.setup_custom_model(
                config_dict, experiment_folder, model_name, ref_model_dict, param_ranges, scorer_func, x_test, y_test
            )
        # Random search
        if hyper_tuning == "random" and not single_model_flag:
            print("Using random search")
            # Do a random search to find the best parameters
            trained_model = random_search(model, model_name, param_ranges, hyper_budget, x_train, y_train, seed_num, scorer_dict, fit_scorer)
            print("=================== Best model from random search: " + model_name + " ====================")
            print(trained_model)
            print("==================================================================")

        # No hyperparameter tuning (and/or the MLPEnsemble is to be run once)
        elif hyper_tuning is None or single_model_flag:
            if hyper_budget is not None:
                print(f"Hyperparameter tuning budget ({hyper_budget}) is not used without tuning")            
            # No tuning, just use the parameters supplied
            trained_model = single_model(model, param_ranges, x_train, y_train, seed_num)

        # Grid search
        elif hyper_tuning == "grid":
            print("Using grid search")
            if hyper_budget is not None:
                print(f"Hyperparameter tuning budget ({hyper_budget}) is not used in a grid search")
            trained_model = grid_search(model, model_name, param_ranges, x_train, y_train, seed_num, scorer_dict, fit_scorer)
            print("=================== Best model from grid search: " + model_name + " ====================")
            print(trained_model)
            print("==================================================================")

        # Save the best model found
        save_model(experiment_folder, trained_model, model_name)

        # Evaluate the best model using all the scores and CV
        performance_results_dict, predictions = evaluate_model(trained_model, config_dict['problem_type'], x_train, y_train, x_test, y_test)
        predictions.to_csv(results_folder/f'{model_name}_predictions.csv',index=False)
        
        # Save the results
        df_performance_results, fname_perfResults = save_results(
            results_folder, df_performance_results, performance_results_dict,
            model_name, fname,
            suffix="_performance_results_testset", save_pkl=False, save_csv=True)
        
        
        print(f"{model_name} complete! Results saved at {Path(fname_perfResults).parents[0]}")

        # Save predictions and probabilities on test set
        #pred_test = trained_model.predict(x_test)
        #prob_test_set = trained_model.predict_proba(x_test)

        #df_test_predictions = pd.DataFrame(list(zip(pred_test, prob_test_set)),
        #                                  columns=['Predictions_TestSet', 'Probabilities_TestSet'])

        #file_name = f"{experiment_folder / 'results' / 'predictions_test_set'}_{model_name}"
        #df_test_predictions.to_csv(file_name + ".csv")


        # # Old version
        #
        # Evaluate the model using the specified scores (on the training data) - previous version
        # scorer_dict_train = eval_scores(config_dict['problem_type'],
        #     scorer_dict, trained_model,
        #     x_train, y_train
        # )
        # # Save the results
        # df_train, _ = save_results(
        #     results_folder, df_train, scorer_dict_train,
        #     model_name, fname,
        #     suffix="_train", save_pkl=False, save_csv=True)
        #
        # # Evaluate on the test data
        # scorer_dict_test = eval_scores(config_dict['problem_type'],
        #     scorer_dict, trained_model,
        #     x_test, y_test
        # )
        # # Save the results
        # df_test, fname_test = save_results(
        #     results_folder, df_test, scorer_dict_test,
        #     model_name, fname,
        #     suffix="_test", save_pkl=False, save_csv=True)



