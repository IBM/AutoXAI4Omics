# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Union, Literal
from typing_extensions import Annotated
from pydantic import BaseModel, PositiveInt, NonNegativeFloat, Field

from models.model_defs import MODELS
from metrics.metric_defs import METRICS
from utils.ml.feature_selection_defs import FS_METHODS, FS_KBEST_METRICS
from utils.vars import CLASSIFICATION, REGRESSION

MODEL_NAMES_ALL = tuple(set().union(*MODELS.values()))
METRICS_NAMES_ALL = tuple(set().union(*METRICS.values()))
FS_NAMES_MENTHODS = tuple(FS_METHODS.keys())
FS_NAMES_KBMETRICS = tuple(FS_KBEST_METRICS)


class AutoModel(BaseModel):
    min_features: Annotated[
        PositiveInt, Field(description="The minimium number of features to consider.")
    ] = 10
    max_features: Annotated[
        Union[PositiveInt, None],
        Field(
            description="The maximum number of features to consider, if None will default to the number of columns in the given dataset."
        ),
    ] = None
    interval: Annotated[
        PositiveInt,
        Field(
            description="The size of the logarithmic increments to consider when searching for the best number of features."
        ),
    ] = 1
    eval_model: Annotated[
        Union[None, Literal[MODEL_NAMES_ALL]],
        Field(description="The estimator to use to evaluate the selected features."),
    ] = None
    eval_metric: Annotated[
        Union[None, Literal[METRICS_NAMES_ALL]],
        Field(
            description="The metric to use to evaluate the model trained on the selected features."
        ),
    ] = None
    low: Annotated[
        bool,
        Field(
            description="A bool to indicate if the lower the eval_metric the better."
        ),
    ] = True

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
    name: Annotated[
        Literal[FS_NAMES_MENTHODS],
        Field(description="The feature selection method to use"),
    ] = "SelectKBest"
    metric: Annotated[
        Union[None, Literal[FS_NAMES_KBMETRICS]],
        Field(
            description="The metric to use during the feature selection, if required."
        ),
    ] = None
    estimator: Annotated[
        Union[None, Literal[MODEL_NAMES_ALL]],
        Field(description="the model to use during the feature selection if required."),
    ] = None

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
    k: Annotated[
        Union[PositiveInt, Literal["auto"]],
        Field(
            description='The number of features to select, if "auto" is chosen it will find the best number of features to use'
        ),
    ] = "auto"
    var_threshold: Annotated[
        NonNegativeFloat,
        Field(description="The value to use for variance thresholding."),
    ] = 0
    auto: Annotated[
        Union[None, AutoModel],
        Field(
            description="The setting for configuring the automated feature selection"
        ),
    ] = AutoModel()
    method: Annotated[
        Union[None, MethodModel],
        Field(
            description="The setting for the method to use for the feature selection."
        ),
    ] = MethodModel()

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
