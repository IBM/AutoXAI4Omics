######
# LIMITATION --- for now only using metrics that take y_true and y_predict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    explained_variance_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    matthews_corrcoef,
    mean_absolute_error,
    # mean_gamma_deviance,
    # mean_poisson_deviance,
    mean_squared_error,
    # mean_squared_log_error,
    # mean_tweedie_deviance,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    zero_one_loss,
    mean_absolute_percentage_error,
    make_scorer,
)

from utils.vars import CLASSIFICATION, REGRESSION


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


METRICS = {
    CLASSIFICATION: {
        "accuracy_score": make_scorer(accuracy_score, greater_is_better=True),
        "f1_score": make_scorer(f1_score, greater_is_better=True, average="weighted"),
        "hamming_loss": make_scorer(hamming_loss, greater_is_better=False),
        "jaccard_score": make_scorer(jaccard_score, greater_is_better=True, average="weighted"),
        "matthews_corrcoef": make_scorer(matthews_corrcoef, greater_is_better=True),
        "precision_score": make_scorer(precision_score, greater_is_better=True, average="weighted"),
        "recall_score": make_scorer(recall_score, greater_is_better=True, average="weighted"),
        "zero_one_loss": make_scorer(zero_one_loss, greater_is_better=False),
        "roc_auc_score": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True, multi_class="ovo"),
        # confusion matrix not included here as it can not be used to optimise models, but it is calculate regardless
        # in metrics:evaluate_model
    },
    REGRESSION: {
        "explained_variance_score": make_scorer(explained_variance_score, greater_is_better=True),
        "mean_squared_error": make_scorer(mean_squared_error, greater_is_better=False),
        # "mean_squared_log_error": make_scorer(mean_squared_log_error, greater_is_better=False),
        "rmse": make_scorer(rmse, greater_is_better=False),
        "mean_absolute_error": make_scorer(mean_absolute_error, greater_is_better=False),
        "median_absolute_error": make_scorer(median_absolute_error, greater_is_better=False),
        "mean_absolute_percentage_error": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
        "r2_score": make_scorer(r2_score, greater_is_better=True),
        # "mean_poisson_deviance": make_scorer(mean_poisson_deviance, greater_is_better=False),
        # "mean_gamma_deviance": make_scorer(mean_gamma_deviance, greater_is_better=False),
        # "mean_tweedie_deviance": make_scorer(mean_tweedie_deviance, greater_is_better=False),
    },
}

# NOTES:
# The following have been removed/commented out for now as they can only work with non negative y values:
#           - mean_squared_log_error
#           - mean_tweedie_deviance
#           - mean_poisson_deviance
#           - mean_gamma_deviance
