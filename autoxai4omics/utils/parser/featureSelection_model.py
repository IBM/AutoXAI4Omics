# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Union, Literal
from pydantic import (
    BaseModel,
    PositiveInt,
    NonNegativeFloat,
)

from models.model_defs import MODELS
from metrics.metric_defs import METRICS
from utils.ml.feature_selection_defs import FS_METHODS, FS_KBEST_METRICS
from utils.vars import CLASSIFICATION, REGRESSION

MODEL_NAMES_ALL = tuple(set().union(*MODELS.values()))
METRICS_NAMES_ALL = tuple(set().union(*METRICS.values()))
FS_NAMES_MENTHODS = tuple(FS_METHODS.keys())
FS_NAMES_KBMETRICS = tuple(FS_KBEST_METRICS)


class AutoModel(BaseModel):
    min_features: PositiveInt = 10
    max_features: Union[PositiveInt, None] = None
    interval: PositiveInt = 1
    eval_model: Union[None, Literal[MODEL_NAMES_ALL]] = None
    eval_metric: Union[None, Literal[METRICS_NAMES_ALL]] = None
    low: bool = True

    def validateWithProblemType(self, problemType):
        if problemType not in [CLASSIFICATION, REGRESSION]:
            # make sure problemType is valid value
            raise ValueError(
                f"problemType must be one of '{CLASSIFICATION}' or '{REGRESSION}'"
            )

        if self.eval_model is None:
            # set evaluate_model if not set
            self.eval_model = (
                "RandomForestClassifier"
                if problemType == CLASSIFICATION
                else "RandomForestRegressor"
            )
        else:
            # if evaluate model is set check it is valid for problem type
            if self.eval_model not in list(MODELS[problemType].keys()):
                raise ValueError(
                    f"{self.eval_model} is not available for {problemType} problems. "
                    f"Please chose from {','.join(list(MODELS[problemType].keys()))}"
                )

        if self.eval_metric is None:
            # set eval metric if not set
            self.eval_metric = (
                "f1_score" if problemType == CLASSIFICATION else "mean_squared_error"
            )
        else:
            # if eval metric is set check it is valid for problem type
            if self.eval_metric not in list(METRICS[problemType].keys()):
                raise ValueError(
                    f"{self.eval_metric} is not available for {problemType} problems. "
                    f"Please choose from {','.join(list(METRICS[problemType].keys()))}"
                )

        self.low = METRICS[problemType][self.eval_metric] == -1


class MethodModel(BaseModel):
    name: Literal[FS_NAMES_MENTHODS] = "SelectKBest"
    metric: Union[None, Literal[FS_NAMES_KBMETRICS]] = None
    estimator: Union[None, Literal[MODEL_NAMES_ALL]] = None

    def validateWithProblemType(self, problemType):
        if problemType not in [CLASSIFICATION, REGRESSION]:
            # make sure problemType is valid value
            raise ValueError(
                f"problemType must be one of '{CLASSIFICATION}' or '{REGRESSION}'"
            )

        if self.name == "SelectKBest":
            if self.metric is None:
                self.metric = (
                    "f_classif" if problemType == CLASSIFICATION else "f_regression"
                )

            if problemType[:7] not in self.metric:
                # TODO: refactor FS_NAMES_KBMETRICS to be separated by problemtype
                raise ValueError(
                    f"Metric '{self.metric}' is not appropriate for a {problemType} problem "
                )

        if self.name == "RFE":
            if self.estimator is None:
                self.estimator = (
                    "RandomForestClassifier"
                    if problemType == CLASSIFICATION
                    else "RandomForestRegressor"
                )

            if self.estimator not in MODELS[problemType].keys():
                raise ValueError(
                    f"{self.estimator} is not appropriate for problem type {problemType}."
                )


class FeatureSelectionModel(BaseModel):
    k: Union[PositiveInt, Literal["auto"]] = "auto"
    var_threshold: NonNegativeFloat = 0
    auto: Union[None, AutoModel] = AutoModel()
    method: Union[None, MethodModel] = MethodModel()

    # TODO: do conditional validation

    def validateWithProblemType(self, problemType):
        # TODO: trigger at higher levels
        if problemType not in [CLASSIFICATION, REGRESSION]:
            # make sure problemType is valid value
            raise ValueError(
                f"problemType must be one of '{CLASSIFICATION}' or '{REGRESSION}'"
            )

        if self.method is not None:
            self.method.validateWithProblemType(problemType=problemType)
            if self.method.name == "RFE":
                self.auto.eval_model = self.method.estimator

        if self.auto is not None:
            self.auto.validateWithProblemType(problemType=problemType)
