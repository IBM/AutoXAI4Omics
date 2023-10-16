from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LinearRegression,
    Ridge,
    RidgeCV,
    SGDRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from models.custom_model import (
    AutoKeras,
    AutoLGBM,
    AutoSKLearn,
    AutoXGBoost,
    FixedKeras,
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import models.model_params as model_params

import logging

omicLogger = logging.getLogger("OmicLogger")


MODELS = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "RidgeCV": RidgeCV,
    "SGDRegressor": SGDRegressor,
    "ElasticNet": ElasticNet,
    "ElasticNetCV": ElasticNetCV,
    "Lars": Lars,
    "LarsCV": LarsCV,
    "Lasso": Lasso,
    "LassoCV": LassoCV,
    "LassoLars": LassoLars,
    "LassoLarsCV": LassoLarsCV,
    "RandomForestRegressor": RandomForestRegressor,
    "KNeighborsRegressor": KNeighborsRegressor,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "MLPClassifier": MLPClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
    "SVC": SVC,
    "GaussianProcessClassifier": GaussianProcessClassifier,
    "RBF": RBF,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "AdaBoostClassifier": AdaBoostClassifier,
}


def select_model_dict(hyper_tuning):
    """
    Select what parameter range specific we are using based on the given hyper_tuning method.
    """
    omicLogger.debug("Get tunning settings...")

    if hyper_tuning == "random":
        ref_model_dict = model_params.sk_random
    elif hyper_tuning == "grid":
        ref_model_dict = model_params.sk_grid
    elif hyper_tuning is None:
        ref_model_dict = model_params.single_model
    else:
        raise ValueError(f"{hyper_tuning} is not a valid option")
    return ref_model_dict


def define_models(problem_type, hyper_tuning):
    """
    Define the models to be run.

    The name is the key, the value is a tuple with the model function, and defined params
    """
    omicLogger.debug("Defining the set of models...")
    ref_model_dict = select_model_dict(hyper_tuning)

    if problem_type == "classification":
        try:
            # Specific modifications for problem type
            if hyper_tuning is None or hyper_tuning == "boaas":
                ref_model_dict["svm"]["probability"] = True
            else:
                ref_model_dict["svm"]["probability"] = [True]
        # Otherwise pass - models may not always be defined for every tuning method
        except KeyError:
            pass
        # Define dict
        model_dict = {
            "rf": (RandomForestClassifier, ref_model_dict["rf"]),
            "svm": (SVC, ref_model_dict["svm"]),
            "knn": (KNeighborsClassifier, ref_model_dict["knn"]),
            "adaboost": (AdaBoostClassifier, ref_model_dict["adaboost"]),
            "xgboost": (XGBClassifier, ref_model_dict["xgboost"]),
        }
    elif problem_type == "regression":
        # Specific modifications for problem type

        # Define dict
        model_dict = {
            "rf": (RandomForestRegressor, ref_model_dict["rf"]),
            "svm": (SVR, ref_model_dict["svm"]),
            "knn": (KNeighborsRegressor, ref_model_dict["knn"]),
            "adaboost": (AdaBoostRegressor, ref_model_dict["adaboost"]),
            "xgboost": (XGBRegressor, ref_model_dict["xgboost"]),
        }
    else:
        raise ValueError(f"{problem_type} is not recognised, must be either 'regression' or 'classification'")
    # The CustomModels handle classification and regression themselves so put outside
    # For mixing tuning types, default to using the single model for mlp_ens

    model_dict["fixedkeras"] = (
        FixedKeras,
        ref_model_dict.get("fixedkeras", model_params.single_model.get("fixedkeras")),
    )

    model_dict["autokeras"] = (
        AutoKeras,
        ref_model_dict.get("autokeras", model_params.single_model.get("autokeras")),
    )

    model_dict["autolgbm"] = (
        AutoLGBM,
        ref_model_dict.get("autolgbm", model_params.single_model.get("autolgbm")),
    )

    model_dict["autoxgboost"] = (
        AutoXGBoost,
        ref_model_dict.get("autoxgboost", model_params.single_model.get("autoxgboost")),
    )

    model_dict["autosklearn"] = (
        AutoSKLearn,
        ref_model_dict.get("autosklearn", model_params.single_model.get("autosklearn")),
    )

    return model_dict
