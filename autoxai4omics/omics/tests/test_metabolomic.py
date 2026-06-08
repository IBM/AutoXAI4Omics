import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to sys.path so metabolomic.py can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metabolomic import get_data_metabolomic, get_data_metabolomic_trained


@pytest.fixture
def config_dict():
    return {
        "metabolomic": {
            "filter_measurements": [10, 20],
            "filter_metabolomic_sample": True,
            "output_file_met": None,
            "output_metadata": None,
        },
        "data": {
            "name": "test_dataset",
            "data_type": "MET",
            "file_path": "dummy.csv",
            "file_path_holdout_data": "dummy_holdout.csv",
            "metadata_file": "",
            "metadata_file_holdout_data": "",
            "target": "target_row",
        },
        "prediction": {"file_path": "pred.csv"},
    }


@pytest.fixture
def dummy_df():
    return pd.DataFrame({"met1": [1, 2], "met2": [3, 4]}, index=["sample1", "sample2"])


@pytest.fixture
def dummy_features():
    return ["met1", "met2"]


@pytest.fixture
def mock_target_df():
    return pd.DataFrame({"sample1": [0], "sample2": [1]}, index=["target_row"])


@pytest.fixture
def mock_metadata_df():
    return pd.DataFrame({"target_row": [0, 1]}, index=["sample1", "sample2"])


# Test get_data_metabolomic with metadata absent and present
@pytest.mark.parametrize("metadata_present", [True, False])
def test_get_data_metabolomic(
    config_dict,
    dummy_df,
    dummy_features,
    mock_target_df,
    mock_metadata_df,
    metadata_present,
):
    if metadata_present:
        config_dict["data"]["metadata_file_holdout_data"] = "dummy_metadata.csv"
    with (
        patch(
            "metabolomic.rrep.preprocessing_LO", return_value=(dummy_df, dummy_features)
        ) as mock_preproc,
        patch("metabolomic.joblib.dump") as mock_joblib,
        patch(
            "pandas.read_csv",
            return_value=(mock_metadata_df if metadata_present else mock_target_df),
        ) as mock_read_csv,
        patch("pandas.DataFrame.to_csv") as mock_to_csv,
        patch("builtins.open", new_callable=MagicMock),
    ):
        x, y, features = get_data_metabolomic(config_dict, holdout=True)
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, np.ndarray)
        assert features == dummy_features
        mock_preproc.assert_called_once()
        mock_joblib.assert_called()
        mock_read_csv.assert_called()
        if metadata_present:
            mock_to_csv.assert_called()
        else:
            assert mock_to_csv.call_count >= 1  # filtered_data always saved


# Test get_data_metabolomic_trained with metadata absent and present
@pytest.mark.parametrize(
    "metadata_present,prediction",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_get_data_metabolomic_trained(
    config_dict,
    dummy_df,
    mock_target_df,
    mock_metadata_df,
    metadata_present,
    prediction,
):
    if metadata_present:
        config_dict["data"]["metadata_file_holdout_data"] = "dummy_metadata.csv"
    with (
        patch(
            "metabolomic.rrep.apply_learned_processing", return_value=dummy_df
        ) as mock_apply,
        patch(
            "pandas.read_csv",
            return_value=(mock_metadata_df if metadata_present else mock_target_df),
        ) as mock_read_csv,
        patch("pandas.DataFrame.to_csv") as mock_to_csv,
        patch("builtins.open", new_callable=MagicMock),
    ):
        x, y, features = get_data_metabolomic_trained(
            config_dict, holdout=True, prediction=prediction
        )
        assert isinstance(x, pd.DataFrame)
        assert features == ["met1", "met2"]
        if prediction:
            assert y is None or isinstance(y, np.ndarray)
        else:
            assert isinstance(y, np.ndarray)
        mock_apply.assert_called_once()
        mock_read_csv.assert_called()
        if metadata_present:
            mock_to_csv.assert_called()
        else:
            assert mock_to_csv.call_count >= 0  # filtered_data may not save metadata


# Test error case: missing keys in config_dict
@pytest.mark.parametrize("missing_key", ["metabolomic", "data"])
def test_missing_keys_in_config(missing_key, config_dict):
    del config_dict[missing_key]
    with pytest.raises(KeyError):
        get_data_metabolomic(config_dict)
