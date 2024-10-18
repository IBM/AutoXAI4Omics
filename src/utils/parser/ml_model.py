# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from typing import Literal, Union, List
from ..vars import REGRESSION, CLASSIFICATION
from metrics.metric_defs import METRICS
from models.model_defs import MODELS
from pydantic import BaseModel, NonNegativeInt, confloat, model_validator
from .featureSelection_model import FeatureSelectionModel
from .autokeras_model import AutoKerasModel
from .autolgbm_model import AutoLgbmModel
from .autoxgboost_model import AutoXgboostModel

TestSize = confloat(strict=True, le=1, ge=0)
METRICS_NAMES_ALL = tuple(set().union(*METRICS.values()))
MODEL_NAMES_ALL = tuple(set().union(*MODELS.values()))


class MlModel(BaseModel):
    seed_num: NonNegativeInt = 29292
    test_size: TestSize = 0.2  # type: ignore
    problem_type: Literal[CLASSIFICATION, REGRESSION]
    hyper_tuning: Literal["random", "grid"] = "random"
    hyper_budget: NonNegativeInt = 50
    stratify_by_groups: Literal["Y", "N"] = "N"
    groups: str = None  # need to check
    balancing: Literal["OVER", "UNDER", "NONE"] = "NONE"
    fit_scorer: Union[None, Literal[METRICS_NAMES_ALL]] = None
    scorer_list: Union[None, List[Literal[METRICS_NAMES_ALL]]] = []
    model_list: List[Literal[MODEL_NAMES_ALL]]
    encoding: Literal["label", "onehot", None] = None
    autokeras_config: Union[AutoKerasModel, None] = AutoKerasModel()
    autolgbm_config: Union[AutoLgbmModel, None] = AutoLgbmModel()
    autoxgboost_config: Union[AutoXgboostModel, None] = AutoXgboostModel()
    feature_selection: Union[FeatureSelectionModel, None] = FeatureSelectionModel()

    @model_validator(mode="after")
    def check(self):
        if self.hyper_tuning == "grid":
            self.hyper_budget = None

        if self.fit_scorer is None:
            self.fit_scorer = (
                "f1_score"
                if self.problem_type == CLASSIFICATION
                else "mean_absolute_percentage_error"
            )
        elif self.fit_scorer not in list(METRICS[self.problem_type].keys()):
            raise ValueError(
                f"fit_scorer must be one of: {list(METRICS[self.problem_type].keys())}. provided: {self.fit_scorer}"
            )

        if (self.scorer_list is None) or (self.scorer_list == []):
            self.scorer_list = [self.fit_scorer]
        elif not set(self.scorer_list).issubset(METRICS[self.problem_type].keys()):
            raise ValueError(
                f"Non-valid options for scorer_list: {set(self.scorer_list)-set(METRICS[self.problem_type].keys())}. ",
                f"Valid options: {set(METRICS[self.problem_type].keys())}",
            )

        if not set(self.model_list).issubset(
            set(MODELS[self.problem_type].keys()).union(set(MODELS["both"].keys()))
        ):
            raise ValueError(
                f"Non-valid options for model_list: {set(self.model_list)-set(MODELS[self.problem_type].keys())}. ",
                f"Valid options: {set(MODELS[self.problem_type].keys())}",
            )

        if self.problem_type == REGRESSION:
            self.encoding = None

        if "AutoKeras" not in self.model_list:
            self.autokeras_config = None
        if "AutoLGBM" not in self.model_list:
            self.autolgbm_config = None
        if "AutoXGBoost" not in self.model_list:
            self.autoxgboost_config = None

        if self.feature_selection:
            self.feature_selection.validateWithProblemType(self.problem_type)

        return self
