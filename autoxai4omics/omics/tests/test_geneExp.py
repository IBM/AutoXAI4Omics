import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geneExp import get_data_gene_expression, get_data_gene_expression_trained

@pytest.fixture
def config_dict():
    return {
        "gene_expression": {
            "filter_genes": [10, 20],
            "filter_sample": True,
            "output_file_ge": None,
            "output_metadata": None,
            "expression_type": "COUNTS"
        },
        "data": {
            "name": "test_dataset",
            "data_type": "RNA",
            "file_path": "dummy.csv",
            "file_path_holdout_data": "dummy_holdout.csv",
            "metadata_file": "",
            "metadata_file_holdout_data": "",
            "target": "target_row"
        },
        "prediction": {
            "file_path": "pred.csv"
        }
    }

@pytest.fixture
def dummy_df():
    return pd.DataFrame({"gene1": [1, 2], "gene2": [3, 4]}, index=["sample1", "sample2"])

@pytest.fixture
def dummy_genes():
    return ["gene1", "gene2"]

# Helper mock for read_csv when target is in data file
@pytest.fixture
def mock_target_df():
    return pd.DataFrame({"sample1": [0], "sample2": [1]}, index=["target_row"])

# Test multiple expression types
@pytest.mark.parametrize("expression_type,preprocessing_func", [
    ("COUNTS", "preprocessing_TMM"),
    ("FPKM", "preprocessing_others"),
    ("Log2FC", "preprocessing_LO")
])
def test_get_data_gene_expression_types(config_dict, dummy_df, dummy_genes, mock_target_df, expression_type, preprocessing_func):
    config_dict["gene_expression"]["expression_type"] = expression_type
    with patch(f"geneExp.rrep.{preprocessing_func}", return_value=(dummy_df, dummy_genes)) as mock_preproc,          patch("geneExp.joblib.dump") as mock_joblib,          patch("pandas.read_csv", return_value=mock_target_df) as mock_read_csv,          patch("pandas.DataFrame.to_csv") as mock_to_csv,          patch("builtins.open", new_callable=MagicMock):
        x, y, features = get_data_gene_expression(config_dict)
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, np.ndarray)
        assert features == dummy_genes
        assert x.equals(dummy_df)
        mock_preproc.assert_called_once()
        mock_to_csv.assert_called()
        mock_joblib.assert_called()
        mock_read_csv.assert_called()

# Test holdout flag
@pytest.mark.parametrize("holdout", [True, False])
def test_get_data_gene_expression_holdout(config_dict, dummy_df, dummy_genes, mock_target_df, holdout):
    with patch("geneExp.rrep.preprocessing_TMM", return_value=(dummy_df, dummy_genes)),          patch("geneExp.joblib.dump"),          patch("pandas.read_csv", return_value=mock_target_df),          patch("pandas.DataFrame.to_csv"),          patch("builtins.open", new_callable=MagicMock):
        x, y, features = get_data_gene_expression(config_dict, holdout=holdout)
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, np.ndarray)
        assert features == dummy_genes

# Test trained function with prediction flag
@pytest.mark.parametrize("prediction", [True, False])
def test_get_data_gene_expression_trained(config_dict, dummy_df, mock_target_df, prediction):
    with patch("geneExp.rrep.apply_learned_processing", return_value=dummy_df),          patch("pandas.read_csv", return_value=mock_target_df),          patch("pandas.DataFrame.to_csv"),          patch("builtins.open", new_callable=MagicMock):
        x, y, features = get_data_gene_expression_trained(config_dict, holdout=True, prediction=prediction)
        assert isinstance(x, pd.DataFrame)
        assert features == ["gene1", "gene2"]
        if prediction:
            assert y is None or isinstance(y, np.ndarray)
        else:
            assert isinstance(y, np.ndarray)

# Test error case: missing keys in config_dict
@pytest.mark.parametrize("missing_key", ["gene_expression", "data"])
def test_missing_keys_in_config(missing_key, config_dict):
    del config_dict[missing_key]
    with pytest.raises(KeyError):
        get_data_gene_expression(config_dict)
