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
            "accuracy_score_Train": accuracy_score(y_train, pred_train),
            "accuracy_score_Test": accuracy_score(y_test, pred_test),
            "f1_score_Train": f1_score(y_train, pred_train, average="weighted"),
            "f1_score_Test": f1_score(y_test, pred_test, average="weighted"),
            "f1_score_PerClass_Train": f1_score(y_train, pred_train, average=None),
            "f1_score_PerClass_Test": f1_score(y_test, pred_test, average=None),
            "precision_score_Train": precision_score(y_train, pred_train, average="weighted"),
            "precision_score_Test": precision_score(y_test, pred_test, average="weighted"),
            "precision_score_PerClass_Train": precision_score(y_train, pred_train, average=None),
            "precision_score_PerClass_Test": precision_score(y_test, pred_test, average=None),
            "recall_score_Train": recall_score(y_train, pred_train, average="weighted"),
            "recall_score_Test": recall_score(y_test, pred_test, average="weighted"),
            "recall_score_PerClass_Train": recall_score(y_train, pred_train, average=None),
            "recall_score_PerClass_Test": recall_score(y_test, pred_test, average=None),
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
        # TODO: remove the definition of score_dict and employ the implementation below
        # score_results_dict={}
        # for score_name, scorer in score_dict.items():
        #     score_results_dict[score_name+"_Train"] = scorer._score_func(y_train,pred_train,**scorer._kwargs)
        #     score_results_dict[score_name+"_Test"] = scorer._score_func(y_test,pred_test,**scorer._kwargs)

        score_dict = {
            "mean_squared_error_Train": skm.mean_squared_error(y_train, pred_train),
            "mean_squared_error_Test": skm.mean_squared_error(y_test, pred_test),
            "mean_absolute_error_Train": skm.mean_absolute_error(y_train, pred_train),
            "mean_absolute_error_Test": skm.mean_absolute_error(y_test, pred_test),
            "median_absolute_error_Train": skm.median_absolute_error(y_train, pred_train),
            "median_absolute_error_Test": skm.median_absolute_error(y_test, pred_test),
            "mean_absolute_percentage_error_Train": skm.mean_absolute_percentage_error(y_train, pred_train),
            "mean_absolute_percentage_error_Test": skm.mean_absolute_percentage_error(y_test, pred_test),
            "r2_score_Train": skm.r2_score(y_train, pred_train),
            "r2_score_Test": skm.r2_score(y_test, pred_test)
            # 'CV_F1Scores': cross_val_score(model, x_train, y_train, scoring='mean_ae', cv=5)
        }

        pred_out = pd.DataFrame(pred_out, columns=["Prediction"])

    return score_dict, pred_out
