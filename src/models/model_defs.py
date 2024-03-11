# Copyright 2024 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    Lars,
    Lasso,
    LassoLars,
    LinearRegression,
    Ridge,
    SGDRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from models.custom_model import (
    AutoKeras,
    AutoLGBM,
    AutoXGBoost,
    FixedKeras,
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import models.model_params as model_params
from utils.vars import CLASSIFICATION, REGRESSION
import logging

omicLogger = logging.getLogger("OmicLogger")


MODELS = {
    REGRESSION: {
        "LinearRegression": {
            "model": LinearRegression,
        },
        "Ridge": {
            "model": Ridge,
        },
        "SGDRegressor": {
            "model": SGDRegressor,
        },
        "ElasticNet": {
            "model": ElasticNet,
        },
        "Lars": {
            "model": Lars,
        },
        "Lasso": {
            "model": Lasso,
        },
        "LassoLars": {
            "model": LassoLars,
        },
        "RandomForestRegressor": {
            "model": RandomForestRegressor,
            "random": model_params.sk_random.get("rf"),
            "grid": model_params.sk_grid.get("rf"),
            "single": model_params.single_model.get("rf"),
        },
        "KNeighborsRegressor": {
            "model": KNeighborsRegressor,
            "random": model_params.sk_random.get("knn"),
            "grid": model_params.sk_grid.get("knn"),
            "single": model_params.single_model.get("knn"),
        },
        "DecisionTreeRegressor": {
            "model": DecisionTreeRegressor,
        },
        "GradientBoostingRegressor": {
            "model": GradientBoostingRegressor,
        },
        "AdaBoostRegressor": {
            "model": AdaBoostRegressor,
            "random": model_params.sk_random.get("adaboost"),
            "grid": model_params.sk_grid.get("adaboost"),
            "single": model_params.single_model.get("adaboost"),
        },
        "XGBRegressor": {
            "model": XGBRegressor,
            "random": model_params.sk_random.get("xgboost"),
            "grid": model_params.sk_grid.get("xgboost"),
            "single": model_params.single_model.get("xgboost"),
        },
        "SVR": {
            "model": SVR,
            "random": model_params.sk_random.get("svr"),
            "grid": model_params.sk_grid.get("svr"),
            "single": model_params.single_model.get("svr"),
        },
    },
    CLASSIFICATION: {
        "XGBClassifier": {
            "model": XGBClassifier,
            "random": model_params.sk_random.get("xgboost"),
            "grid": model_params.sk_grid.get("xgboost"),
            "single": model_params.single_model.get("xgboost"),
        },
        "GradientBoostingClassifier": {
            "model": GradientBoostingClassifier,
        },
        "MLPClassifier": {
            "model": MLPClassifier,
        },
        "KNeighborsClassifier": {
            "model": KNeighborsClassifier,
            "random": model_params.sk_random.get("knn"),
            "grid": model_params.sk_grid.get("knn"),
            "single": model_params.single_model.get("knn"),
        },
        "SVC": {
            "model": SVC,
            "random": model_params.sk_random.get("svc"),
            "grid": model_params.sk_grid.get("svc"),
            "single": model_params.single_model.get("svc"),
        },
        "GaussianProcessClassifier": {
            "model": GaussianProcessClassifier,
        },
        "RBF": {
            "model": RBF,
        },
        "DecisionTreeClassifier": {
            "model": DecisionTreeClassifier,
        },
        "RandomForestClassifier": {
            "model": RandomForestClassifier,
            "random": model_params.sk_random.get("rf"),
            "grid": model_params.sk_random.get("rf"),
            "single": model_params.single_model.get("rf"),
        },
        "AdaBoostClassifier": {
            "model": AdaBoostClassifier,
            "random": model_params.sk_random.get("adaboost"),
            "grid": model_params.sk_grid.get("adaboost"),
            "single": model_params.single_model.get("adaboost"),
        },
    },
    "both": {
        "AutoKeras": {
            "model": AutoKeras,
        },
        "AutoLGBM": {
            "model": AutoLGBM,
        },
        "AutoXGBoost": {
            "model": AutoXGBoost,
        },
        "FixedKeras": {
            "model": FixedKeras,
            "single": model_params.single_model.get("FixedKeras"),
        },
    },
}


def form_model_dict(
    problem_type: str, hyper_tunning: [str, None], model_list: list[str]
) -> dict[str, tuple[object, dict, bool]]:
    """
    A function to form a dict with the model names as keys and the nessicary items to instansiate the object.

    Returns
    -------
    dict
        chosen model names are the keys and the values is a tuple containing the model object, the paramaters for the
        hyper tunning (defaulting to single if none found) and a boolean flag indicating if the parameter are for a
        single model

    Raises
    ------
    ValueError
        Is raised if problem type is not 'classification' or 'regression'
    ValueError
        Is raised if hyper_tunning is not one of 'grid', 'random' or None
    ValueError
        Is raised if the model specified in model_name is not available for training
    """

    # check that the problem type is one of the accepted entries
    if problem_type not in [CLASSIFICATION, REGRESSION]:
        raise ValueError(
            f"problem_type must be one of classification or regression, provided: {problem_type}"
        )

    # check that the hyper_tunning is one of the accepted entries
    if hyper_tunning not in ["grid", "random", None]:
        raise ValueError(
            f"hyper_tuning must be one of 'grid', 'random' or None. Provided {hyper_tunning}"
        )

    # if hyper_tunning is none set to 'single' to access appropriate entries
    if hyper_tunning is None:
        hyper_tunning = "single"

    # create empty out dict
    model_dict = {}

    # combine problem specific models and flexible models into one dict
    combi = {**MODELS[problem_type], **MODELS["both"]}

    # for each model name in our model list
    for model_name in model_list:
        # check that the model name is valid
        if model_name not in combi.keys():
            raise ValueError(
                f"{model_name} is not available for the given problem type ({problem_type}). Available "
                f"models are {','.join(combi.keys())}"
            )

        # get the model object
        mdl = combi[model_name]["model"]

        # get the parameters for the specified hyper tunning, will be none if does not exist
        hyper_tunning_params = combi[model_name].get(hyper_tunning)

        # get the single model params as the backup and default to empty dict if none found
        single_tunning_params = combi[model_name].get("single", {})

        # if the hyper_tunning specified is not none
        if hyper_tunning_params:
            # then return the hyper parameters chosen
            model_dict[model_name] = (mdl, hyper_tunning_params, False)
        else:
            # else return the single model parameters
            model_dict[model_name] = (mdl, single_tunning_params, True)

    return model_dict
