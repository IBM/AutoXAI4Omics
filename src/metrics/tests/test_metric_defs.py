import pytest

from ..metric_defs import METRICS

REG_Y_TRUE = list(range(1, 11))
REG_Y_PRED = REG_Y_TRUE.copy()
REG_Y_PRED.reverse()

CLF_Y_TRUE = [0] * 4 + [1] * 4 + [2] * 2
CLF_Y_PRED = CLF_Y_TRUE.copy()
CLF_Y_PRED.reverse()


RESULTS = {
    "regression": {
        "explained_variance_score": -3.0,
        "mean_squared_error": 33.0,
        "mean_squared_log_error": 1.0515980461430081,
        "rmse": 5.744562646538029,
        "mean_absolute_error": 5.0,
        "median_absolute_error": 5.0,
        "mean_absolute_percentage_error": 1.8014682539682538,
        "r2_score": -3.0,
        "mean_poisson_deviance": 7.603424560007838,
        "mean_gamma_deviance": 2.443730158730159,
        "mean_tweedie_deviance": 33.0,
    },
    "classification": {
        "accuracy_score": 0.2,
        "f1_score": 0.2,
        "hamming_loss": 0.8,
        "jaccard_score": 0.13333333333333333,
        "matthews_corrcoef": -0.25,
        "precision_score": 0.2,
        "recall_score": 0.2,
        "zero_one_loss": 0.8,
    },
}


@pytest.mark.metrics
@pytest.mark.regression
@pytest.mark.parametrize("metric_name", METRICS["regression"].keys())
def test_regression_metrics(metric_name):
    metric_instance = METRICS["regression"][metric_name]
    score = metric_instance._score_func(REG_Y_TRUE, REG_Y_PRED, **metric_instance._kwargs)
    assert score == RESULTS["regression"][metric_name]


@pytest.mark.metrics
@pytest.mark.classification
@pytest.mark.parametrize("metric_name", METRICS["classification"].keys())
def test_classification_metrics(metric_name):
    metric_instance = METRICS["classification"][metric_name]
    score = metric_instance._score_func(CLF_Y_TRUE, CLF_Y_PRED, **metric_instance._kwargs)
    assert score == RESULTS["classification"][metric_name]
