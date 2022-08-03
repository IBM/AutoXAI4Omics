# import pytest
# import sys
# sys.path.append("..")
# import data_processing
# import numpy as np
# pytestmark = pytest.mark.data_processing


# @pytest.mark.transform
# class TestStanardisationFunctions:
    
#     def test_standardize_data():
#         x, _, _ = make_dataset(problem_type='regression',**std_dataset_def_reg)
        
#         data, SS = data_processing.standardize_data(x)
        
#         assert all(data.mean(axis=0).round(10)==0)&all(data.std(axis=0).round(10)==1)
        