import subprocess
import pytest
# import tensorflow
from os.path import exists
# import joblib
# import numpy as np
import sys
sys.path.append('../auto-omics/')
# import custom_model
# import optuna
import yaml
import pandas as pd
    
@pytest.mark.scripts
@pytest.mark.parametrize("script", [
    pytest.param('./train_models.sh', marks=pytest.mark.training), 
    pytest.param('./testing_holdout.sh', marks=pytest.mark.holdout), 
    pytest.param('./plotting.sh', marks=pytest.mark.plotting)
    ])
def test_scripts(script,problem_create):
    fname = problem_create.split('/')[1]
    sp = subprocess.call([script, fname])
    assert sp==0
    
    
@pytest.mark.output
@pytest.mark.parametrize("problem",[
    pytest.param('classification', marks=[
        pytest.mark.classification,
        pytest.mark.skipif(exists('/experiments/results/generated_test_classification_run1_1/best_model/'), reason="Best model folder was not created")
        ]),
    pytest.param('multi', marks=[
        pytest.mark.classification,
        pytest.mark.skipif(exists('/experiments/results/generated_test_classification_multi_run1_1/best_model/'), reason="Best model folder was not created")
        ]),
    pytest.param('regression', marks=[
        pytest.mark.regression,
        pytest.mark.skipif(exists('/experiments/results/generated_test_regression_run1_1/best_model/'), reason="Best model folder was not created")
        ])
    ])
def test_model_outputs(problem):
    with open('tests/result_sets/best_model_names.yml') as file:
        best_model_names = yaml.safe_load(file)
    
    if problem != 'multi':
        assert exists(f'experiments/results/generated_test_{problem}_run1_1/best_model/{best_model_names[problem]}_best.pkl')
        
        df_run = pd.read_csv(f'experiments/results/generated_test_{problem}_run1_1/results/scores__performance_results_testset.csv').set_index('model')
        df_stored = pd.read_csv(f'tests/result_sets/{problem}_results.csv').set_index('model')
        
        assert (df_run==df_stored).all().all()
        
    else:
        assert exists(f'experiments/results/generated_test_classification_{problem}_run1_1/best_model/{best_model_names[problem]}_best.pkl')
        
        df_run = pd.read_csv(f'experiments/results/generated_test_classification_{problem}_run1_1/results/scores__performance_results_testset.csv').set_index('model')
        df_stored = pd.read_csv(f'tests/result_sets/{problem}_results.csv').set_index('model')
        
        assert (df_run==df_stored).all().all()
        