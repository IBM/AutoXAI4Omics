import numpy as np
import pandas as pd
from sklearn.metrics._scorer import _ProbaScorer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from metrics.metric_defs import METRICS
import logging
from utils.vars import CLASSIFICATION, REGRESSION

omicLogger = logging.getLogger("OmicLogger")


def define_scorers(problem_type: str, scorer_list: list[str]) -> dict[str, object]:
    """
    Create a dict with the keywords being the scorers we can use and the values being the object itself


    Parameters
    ----------
    problem_type : str
        Either 'classification' or 'regression'
    scorer_list : list[str]
        Must be a list of str corresponding to names of scorers that the user wishes to be calculated

    Returns
    -------
    dict[str, object]
        a dict with the scorer names as keys and the scoring function/object as the value

    Raises
    ------
    TypeError
        is raised if problem_type is not a str
    ValueError
        is raised if problem_type is not one of 'classification' or 'regression'
    TypeError
        is raised if scorer_list is not a str
    ValueError
        is raised if scorer_list is empty
    TypeError
        is raised if any entry of scorer_list is not a str
    ValueError
        is raised if any entry of scorer_list is not a valid scorer name
    """
    if not isinstance(problem_type, str):
        raise TypeError("problem_type must be of type str")
    elif problem_type not in [CLASSIFICATION, REGRESSION]:
        raise ValueError(f"problem_type must be one of {CLASSIFICATION} or {REGRESSION}")

    if not isinstance(scorer_list, list):
        raise TypeError("scorer_list must be a list")
    elif len(scorer_list) < 1:
        raise ValueError("scorer_list can not be empty")
    elif not all([isinstance(x, str) for x in scorer_list]):
        raise TypeError("all entries of scorer_list must be a str")

    omicLogger.debug("Collecting model scorers specified in config...")
    try:
        scorer_dict_filtered = {k: METRICS[problem_type][k] for k in scorer_list}
    except KeyError as e:
        raise ValueError(
            f"For {problem_type} problems, entries of scorer_list must be a subset of: \
                         {METRICS[problem_type].keys()}, recived invalid entry of: {e}"
        )

    return scorer_dict_filtered


## TODO: finish creating test for these, other tests need to be done first
def eval_scores(problem_type, scorer_dict, model, data, true_labels):
    omicLogger.debug("Gathering evaluation scores...")
    scores_dict = {}
    for score_name, score_func in scorer_dict.items():
        if problem_type == REGRESSION:
            scores_dict[score_name] = np.abs(score_func(model, data, true_labels))
        else:
            scores_dict[score_name] = score_func(model, data, true_labels)

    return scores_dict


def evaluate_model(model, problem_type, x_train, y_train, x_test, y_test, score_dict):
    """
    Define the different measures we can use and Calculate some of them on the model
    """
    omicLogger.debug("Evaluate the model...")

    pred_test = model.predict(x_test)
    pred_train = model.predict(x_train)
    pred_out = np.concatenate((pred_train, pred_test))

    if problem_type == CLASSIFICATION:
        col_names = ["Prediction"]
        omicLogger.debug("calculating prediction probabilities...")
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

        score_results_dict = {}
        for score_name, scorer in score_dict.items():
            omicLogger.debug(f"Calculating {score_name} for Train and Test sets...")
            if not isinstance(scorer, _ProbaScorer):
                score_results_dict[score_name + "_Train"] = scorer._score_func(y_train, pred_train, **scorer._kwargs)
                score_results_dict[score_name + "_Test"] = scorer._score_func(y_test, pred_test, **scorer._kwargs)
            else:
                score_results_dict[score_name + "_Train"] = scorer._score_func(
                    y_train, pred_train_proba, **scorer._kwargs
                )
                score_results_dict[score_name + "_Test"] = scorer._score_func(y_test, pred_test_proba, **scorer._kwargs)

            if score_name in ["f1_score", "precision_score", "recall_score"]:
                omicLogger.debug(f"Calculating {score_name} perClass for Train and Test sets...")
                score_results_dict[score_name + "_PerClass_Train"] = scorer._score_func(
                    y_train, pred_train, average=None
                )
                score_results_dict[score_name + "_PerClass_Test"] = scorer._score_func(y_test, pred_test, average=None)

        # confusion matrix doesnt return a single numeric value and can not be used for optimisation, but is good
        # to report hence why its defined here and not in metric_defs.py
        omicLogger.debug("Calculating confusion matrix...")
        score_results_dict["confusion_matrix_Train"] = confusion_matrix(y_train, pred_train)
        score_results_dict["confusion_matrix_Test"] = confusion_matrix(y_test, pred_test)

        pred_out = pd.DataFrame(
            np.concatenate((pred_out.reshape(-1, 1), prob_out), axis=1),
            columns=col_names,
        )
    else:
        score_results_dict = {}
        for score_name, scorer in score_dict.items():
            omicLogger.debug(f"Calculating {score_name} for Train and Test sets...")
            score_results_dict[score_name + "_Train"] = scorer._score_func(y_train, pred_train, **scorer._kwargs)
            score_results_dict[score_name + "_Test"] = scorer._score_func(y_test, pred_test, **scorer._kwargs)

        pred_out = pd.DataFrame(pred_out, columns=["Prediction"])

    return score_results_dict, pred_out
