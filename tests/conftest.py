import pytest
from sklearn.datasets import make_regression, make_classification
import pandas as pd
import os
import json
import shutil

##################### DATA SET CREATION #####################

def dataset_create_data(problem_type,**kwargs):
    if problem_type == 'regression':
        x, y, coef = make_regression(**kwargs)
        return x, y, coef
    
    elif problem_type == 'classification':
        x, y = make_classification(**kwargs) 
        return x, y 
    
    else:
        raise ValueError(f"problem_type is not regression or classification. value given: {problem_type}")
        

def dataset_std_def_reg():
    def_dict = {
        'n_samples' : 100,
        'n_features' : 20,
        'n_informative' : 10,
        'n_targets' : 1,
        'shuffle' : True,
        'coef' : True,
        'random_state' : 29292
    }
    return def_dict


def dataset_std_def_clf_bin():
    def_dict = {'n_samples' : 100,
                'n_features' : 20,
                'n_informative' : 10,
                'n_classes' : 2,
                'n_clusters_per_class' : 1,
                'weights' : [0.6],
                'flip_y' : 0,
                'shuffle' : True,
                'random_state' : 29292
    }
    return def_dict


def dataset_std_def_clf_mult():
    def_dict = dataset_std_def_clf_bin()
    def_dict['n_classes'] = 3 
    def_dict['weights'] = [0.5,0.3]
    return def_dict


def dataset_create_files(problem_type,multi=False):
    if problem_type == 'regression':
        x, y, _ = make_regression(**dataset_std_def_reg())
    elif problem_type == 'classification':
        if multi:
            x, y = make_classification(**dataset_std_def_clf_mult()) 
        else:
            x, y = make_classification(**dataset_std_def_clf_bin())
    else:
        raise ValueError(f"problem_type is not regression or classification. value given: {problem_type}")
    
    x_fname = f"data/generated_training_{problem_type}"
    y_fname = f"data/generated_target_{problem_type}"
    
    x_fname += '_multi.csv' if (multi and (problem_type=='classification')) else '.csv'
    y_fname += '_multi.csv' if (multi and (problem_type=='classification')) else '.csv'
    
    pd.DataFrame(x).to_csv(x_fname)
    pd.DataFrame(y,columns=['target']).to_csv(y_fname)
    
    return x_fname, y_fname 
    
##################### CONFIG CREATION #####################

def config_autosklearn():
    outdict = {
        "autosklearn_config": {
            "verbose": True,
            "estimators": [
                "decision_tree",
                "extra_trees",
                "k_nearest_neighbors",
                "random_forest",
            ],
            "time_left_for_this_task": 120,
            "per_run_time_limit": 60,
            "memory_limit": None,
            "n_jobs": 1,
            "ensemble_size": 1
        }
    }
    return outdict

def config_autokeras():
    outdict = {
        "autokeras_config": {
            "n_epochs": 100,
            "batch_size": 32,
            "verbose": True,
            "n_blocks": 3,
            "dropout": 0.3,
            "use_batchnorm": True,
            "n_trials": 4,
            "tuner": "bayesian"
        }
    }
    return outdict

def config_autolgbm():
    outdict = {
        "autolgbm_config": {
            "verbose": True,
            "n_trials": 15,
            "timeout": 60
        },
    }
    return outdict

def config_autoxgboost():
    outdict = {
        "autoxgboost_config": {
            "verbose": True,
            "n_trials": 10,
            "timeout": 1000
        },
    }
    return outdict

# def config_autogluon():
#     outdict = {
#         "autogluon_config": {
#             "verbose": True,
#             "auto_stack": False,
#             "time_limits": 120
#         },
#     }
#     return outdict

def config_all_auto_methods():
    outdict = {
        **config_autosklearn(),
        **config_autokeras(),
        **config_autolgbm(),
        **config_autoxgboost()
    }
    return outdict


def config_plot_classification():
    outlist = ["conf_matrix","roc_curve"]
    return outlist


def config_plot_regression():
    outlist = ['hist_overlapped','joint','joint_dens','corr',]
    return outlist


def config_plot_both():
    outlist = ['barplot_scorer','boxplot_scorer','shap_plots','permut_imp_test',]
    return outlist


def config_all_plotting(problem_type):
    outlist = config_plot_both()
    if problem_type =='regression':
        outlist += config_plot_regression()
    elif problem_type == 'classification':
        outlist += config_plot_classification()
    
    outdict = {
        "plot_method" : outlist,
        "top_feats_permImp": 20,
        "top_feats_shap": 20,
        "explanations_data": "all",
    }
    return outdict


def config_feature_selection():
    outdict = {
        "feature_selection": {
            "k": "auto"
        }
    }
    return outdict


def config_model_list():
    outdict = {
        "model_list": [
            "rf",
            "adaboost",
            "knn",
            "autoxgboost",
            "autolgbm",
            "autosklearn",
            "autokeras",
            "fixedkeras",
        ],
    }
    return outdict


def config_scorers(problem_type):
    if problem_type == 'regression':
        outdict = {
            "fit_scorer": "mean_ape",
            "scorer_list": [
                "mean_ae",
                "med_ae",
                "rmse",
                "mean_ape",
                "r2"
            ],
        }
    elif problem_type == 'classification':
        outdict = {
            "fit_scorer": "f1",
            "scorer_list": [
                "acc",
                "f1"
            ],
        }
    return outdict


def config_data_paths(file_path,meta_path):
    outdict = {
        "file_path": '/'+file_path,
        "metadata_file": '/'+meta_path,
        "file_path_holdout_data": '/'+file_path,
        "metadata_file_holdout_data": '/'+meta_path,
        "save_path": "/experiments/",
        "target": "target",
        "data_type": "test_data",
    }
    return outdict


def config_define_problem(problem_type,multi=False):
    outdict = {
        "name": "generated_test_"+problem_type+ ('_multi' if multi else '')+'_run1_',
        "seed_num": 29292,
        "test_size": 0.2,
        "problem_type": problem_type,
        "hyper_tuning": "random",
        "hyper_budget": 50,
        "stratify_by_groups": "N",
        "groups": "",
        "oversampling": "Y",
    }
    return outdict


def config_microbiome():
    outdict = {
        'collapse_tax':None,
        'min_reads':None,
        'norm_reads':None,
        'filter_abundance':None,
        'filter_prevalence':None,
        'filter_microbiome_samples':None,
        'remove_classes':None,
        'merge_classes':None,
    }
    return outdict


def config_gene_expression():
    outdict = {
        'expression_type':None,
        'filter_sample':None,
        'filter_genes':None,
        'output_file_ge':None,
        'output_metadata':None,
    }
    return outdict


def config_metabolmic():
    outdict = {
        'filter_metabolomic_sample':None,
        'filter_measurements':None,
        'output_file_met':None,
        'output_metadata':None,
    }
    return outdict


def config_tabular():
    outdict = {
        'filter_tabular_sample':None,
        'filter_tabular_measurements':None,
        'output_file_tab':None,
        'output_metadata':None,
    }
    return outdict


def config_create(problem_type,file_path,meta_path,multi=False,run=1):
    outdict = {
        **config_define_problem(problem_type,multi),  # Define the Problem and general parameters            # [EXP/AUTO]
        **config_data_paths(file_path,meta_path),     # Define the paths to where the data is stored         # [EXP/AUTO]
        **config_scorers(problem_type),               # Define scorers to be used depending on problem type  # [HIDDEN]
        **config_model_list(),                        # Define the models to be trained                      # [ADV]
        **config_all_plotting(problem_type),          # Define what plots to do                              # [HIDDEN]
        **config_all_auto_methods(),                  # Define auto methods setting                          # [HIDDEN]
        **config_feature_selection(),                 # Define the feature selection settings                # [ADV/AUTO]
        **config_microbiome(),                        # settings for the corresponding data type             # [?]
        # **config_gene_expression(),                 # settings for the corresponding data type             # [?]
        # **config_metabolmic(),                      # settings for the corresponding data type             # [?]
        # **config_tabular(),                         # settings for the corresponding data type             # [?]
    }
    
    outdict['name'] += str(run).zfill(len(str(RUNS)))
    fname = f"configs/generated_test_{problem_type}"
    fname += "_multi.json" if (multi and (problem_type=='classification')) else ".json"
    
    with open(fname, 'w') as outfile:
        json.dump(outdict, outfile,indent=4)
        
    return fname
    
    
    
##################### TEST PROBLEM CREATION #####################
STARTS=0
RUNS=1
@pytest.fixture(
    params=[pytest.param(('classification',i), marks=pytest.mark.classification) for i in range(STARTS+1,RUNS+1)]+[pytest.param(('multi',i), marks=pytest.mark.classification) for i in range(STARTS+1,RUNS+1)]+[pytest.param(('regression',i), marks=pytest.mark.regression) for i in range(STARTS+1,RUNS+1)], 
    scope="session",
    )
def problem_create(request):
    problem_type = request.param[0]
    run = request.param[1]
    if problem_type == 'multi':
        x_fname, y_fname = dataset_create_files('classification',multi=True)
        fname = config_create('classification',x_fname, y_fname,multi=True, run=run)
    else:
        x_fname, y_fname = dataset_create_files(problem_type)
        fname = config_create(problem_type,x_fname, y_fname, run=run)
        
    yield fname
    
    os.remove(x_fname)
    os.remove(y_fname)
    os.remove(fname)
    # shutil.rmtree(f'experiments/results/{fname[:-5]}_run1_{str(run).zfill(len(str(RUNS)))}')