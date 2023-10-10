import subprocess
import pytest

# import tensorflow
from os.path import exists

# import joblib
# import numpy as np
import sys

# import custom_model
# import optuna
import yaml
import pandas as pd
import json
import glob

sys.path.append("../auto-omics/")


@pytest.mark.modes
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("train", marks=pytest.mark.training),
        pytest.param("test", marks=pytest.mark.holdout),
        pytest.param("plotting", marks=pytest.mark.plotting),
        pytest.param("predict", marks=pytest.mark.prediction),
        pytest.param("feature", marks=pytest.mark.feature),
    ],
)
def test_modes(mode, problem_create):
    fname = problem_create.split("/")[1]
    sp = subprocess.call(["./auto_omics.sh", "-m", mode, "-c", fname])
    assert sp == 0

    with open(fname, "r") as infile:
        config = json.load(infile)
    log_filepath = config["data"]["save_path"][1:] + f'results/{config["data"]["name"]}/AutoOmicLog_*'
    log_filepath = glob.glob(log_filepath)[-1]
    with open(log_filepath, "r") as F:
        last_line = F.readlines()[-1]
    assert "INFO : Process completed." in last_line


@pytest.mark.output
@pytest.mark.parametrize(
    "problem",
    [
        pytest.param(
            "classification",
            marks=[
                pytest.mark.classification,
                pytest.mark.skipif(
                    exists("/experiments/results/generated_test_classification_run1_1/best_model/"),
                    reason="Best model folder was not created",
                ),
            ],
        ),
        pytest.param(
            "multi",
            marks=[
                pytest.mark.classification,
                pytest.mark.skipif(
                    exists("/experiments/results/generated_test_classification_multi_run1_1/best_model/"),
                    reason="Best model folder was not created",
                ),
            ],
        ),
        pytest.param(
            "regression",
            marks=[
                pytest.mark.regression,
                pytest.mark.skipif(
                    exists("/experiments/results/generated_test_regression_run1_1/best_model/"),
                    reason="Best model folder was not created",
                ),
            ],
        ),
    ],
)
def test_model_outputs(problem):
    with open("tests/result_sets/best_model_names.yml") as file:
        best_model_names = yaml.safe_load(file)

    if problem != "multi":
        assert exists(
            f"experiments/results/generated_test_{problem}_run1_1/best_model/{best_model_names[problem]}_best.pkl"
        )

        df_run = pd.read_csv(
            f"experiments/results/generated_test_{problem}_run1_1/results/scores__performance_results_testset.csv"
        ).set_index("model")
        df_stored = pd.read_csv(f"tests/result_sets/{problem}_results.csv").set_index("model")

        assert (df_run == df_stored).all().all()

    else:
        assert exists(
            f"experiments/results/generated_test_classification_{problem}_run1_1/best_model/{best_model_names[problem]}_best.pkl"
        )

        df_run = pd.read_csv(
            f"experiments/results/generated_test_classification_{problem}_run1_1/results/scores__performance_results_testset.csv"
        ).set_index("model")
        df_stored = pd.read_csv(f"tests/result_sets/{problem}_results.csv").set_index("model")

        assert (df_run == df_stored).all().all()


@pytest.mark.omics
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("train", marks=pytest.mark.training),
        pytest.param("test", marks=pytest.mark.holdout),
        pytest.param("plotting", marks=pytest.mark.plotting),
        pytest.param("predict", marks=pytest.mark.prediction),
        pytest.param("feature", marks=pytest.mark.feature),
    ],
)
@pytest.mark.parametrize(
    "omic",
    [
        pytest.param("geneExp", marks=pytest.mark.gene),
        pytest.param("metabolomic", marks=pytest.mark.metabolomic),
        pytest.param("microbiome", marks=pytest.mark.microbiome),
        pytest.param("tabular", marks=pytest.mark.tabular),
    ],
)
@pytest.mark.parametrize(
    "problem",
    [
        pytest.param("binary", marks=pytest.mark.classification),
        pytest.param("multi", marks=pytest.mark.classification),
        pytest.param("reg", marks=pytest.mark.regression),
    ],
)
def test_omic_datasets(mode, omic, problem):
    fname = f"OmicsTestSets/configs/test_{omic}_{problem}.json"
    sp = subprocess.call(["./auto_omics.sh", "-m", mode, "-c", fname])
    assert sp == 0

    with open(fname, "r") as infile:
        config = json.load(infile)
    log_filepath = config["data"]["save_path"][1:] + f'results/{config["data"]["name"]}/AutoOmicLog_*'
    log_filepath = glob.glob(log_filepath)[-1]
    with open(log_filepath, "r") as F:
        last_line = F.readlines()[-1]
    assert "INFO : Process completed." in last_line
