import numpy as np
import pandas as pd
import sklearn.metrics as skm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import normalize
from metrics.metric_defs import METRICS
import logging

omicLogger = logging.getLogger("OmicLogger")


def define_scorers(problem_type: str, scorer_list: list[str]) -> dict[str, object]:
    """
    Define the different measures we can use
    """
    omicLogger.debug("Collecting model scorers specified in config...")
    scorer_dict_filtered = {k: METRICS[problem_type][k] for k in scorer_list}

    return scorer_dict_filtered


def eval_scores(problem_type, scorer_dict, model, data, true_labels):
    omicLogger.debug("Gathering evaluation scores...")
    scores_dict = {}
    for score_name, score_func in scorer_dict.items():
        if problem_type == "regression":
            scores_dict[score_name] = np.abs(score_func(model, data, true_labels))
        else:
            scores_dict[score_name] = score_func(model, data, true_labels)

    return scores_dict


def evaluate_model(model, problem_type, x_train, y_train, x_test, y_test):
    """
    Define the different measures we can use and Calculate some of them on the model
    """
    omicLogger.debug("Evaluate the model...")

    pred_test = model.predict(x_test)
    pred_train = model.predict(x_train)
    pred_out = np.concatenate((pred_train, pred_test))

    if problem_type == "classification":
        col_names = ["Prediction"]
        if len(set(y_train)) == 2:
            pred_test_proba = model.predict_proba(x_test)[:, 1]
            pred_train_proba = model.predict_proba(x_train)[:, 1]
            prob_out = np.concatenate((pred_train_proba, pred_test_proba)).reshape(-1, 1)
            col_names += ["probability"]
        else:
            pred_test_proba = normalize(model.predict_proba(x_test), axis=1, norm="l1")
            pred_train_proba = normalize(model.predict_proba(x_train), axis=1, norm="l1")
            prob_out = np.concatenate((pred_train_proba, pred_test_proba))
            col_names += [f"class_{i}_prob" for i in range(len(set(y_train)))]

        score_dict = {
            "Accuracy_Train": accuracy_score(y_train, pred_train),
            "Accuracy_Test": accuracy_score(y_test, pred_test),
            "F1_score_Train": f1_score(y_train, pred_train, average="weighted"),
            "F1_score_Test": f1_score(y_test, pred_test, average="weighted"),
            "F1_score_PerClass_Train": f1_score(y_train, pred_train, average=None),
            "F1_score_PerClass_Test": f1_score(y_test, pred_test, average=None),
            "Precision_Train": precision_score(y_train, pred_train, average="weighted"),
            "Precision_Test": precision_score(y_test, pred_test, average="weighted"),
            "Precision_PerClass_Train": precision_score(y_train, pred_train, average=None),
            "Precision_PerClass_Test": precision_score(y_test, pred_test, average=None),
            "Recall_Train": recall_score(y_train, pred_train, average="weighted"),
            "Recall_Test": recall_score(y_test, pred_test, average="weighted"),
            "Recall_PerClass_Train": recall_score(y_train, pred_train, average=None),
            "Recall_PerClass_Test": recall_score(y_test, pred_test, average=None),
            "Conf_matrix_Train": confusion_matrix(y_train, pred_train),
            "Conf_matrix_Test": confusion_matrix(y_test, pred_test),
            "ROC_auc_score_Train": roc_auc_score(y_train, pred_train_proba, multi_class="ovo"),
            "ROC_auc_score_Test": roc_auc_score(y_test, pred_test_proba, multi_class="ovo"),
            # 'CV_F1Scores': cross_val_score(model, x_train, y_train, scoring='f1_weighted', cv=5)
        }

        pred_out = pd.DataFrame(
            np.concatenate((pred_out.reshape(-1, 1), prob_out), axis=1),
            columns=col_names,
        )
    else:
        score_dict = {
            "MSE_Train": skm.mean_squared_error(y_train, pred_train),
            "MSE_Test": skm.mean_squared_error(y_test, pred_test),
            "Mean_AE_Train": skm.mean_absolute_error(y_train, pred_train),
            "Mean_AE_Test": skm.mean_absolute_error(y_test, pred_test),
            "Med_ae_Train": skm.median_absolute_error(y_train, pred_train),
            "Med_ae_Test": skm.median_absolute_error(y_test, pred_test),
            "Mean_APE_Train": skm.mean_absolute_percentage_error(y_train, pred_train),
            "Mean_APE_Test": skm.mean_absolute_percentage_error(y_test, pred_test),
            "R2_Train": skm.r2_score(y_train, pred_train),
            "R2_Test": skm.r2_score(y_test, pred_test)
            # 'CV_F1Scores': cross_val_score(model, x_train, y_train, scoring='mean_ae', cv=5)
        }

        pred_out = pd.DataFrame(pred_out, columns=["Prediction"])

    return score_dict, pred_out
