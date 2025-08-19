# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from ..vars import REGRESSION, CLASSIFICATION
from .autokeras_model import AutoKerasModel
from .autolgbm_model import AutoLgbmModel
from .autoxgboost_model import AutoXgboostModel
from .featureSelection_model import FeatureSelectionModel
from metrics.metric_defs import METRICS
from models.model_defs import MODELS
from pydantic import BaseModel, NonNegativeInt, confloat, model_validator, Field
from typing import Literal, Union, List
from typing_extensions import Annotated

TestSize = confloat(strict=True, le=1, ge=0)
METRICS_NAMES_ALL = tuple(set().union(*METRICS.values()))
MODEL_NAMES_ALL = tuple(set().union(*MODELS.values()))


class MlModel(BaseModel):
    seed_num: Annotated[
        NonNegativeInt,
        Field(
            description="The random set to set for this run, used for making the results reproducible."
        ),
    ] = 29292
    test_size: Annotated[TestSize, Field(description="The percentage of the data to use for testing")] = 0.2  # type: ignore
    problem_type: Annotated[
        Literal[CLASSIFICATION, REGRESSION],
        Field(description="The problem type that this job shall be attempting."),
    ]
    # TODO: consider making hyper tuning a submodel
    # TODO: add None to hyper tunning method
    hyper_tuning: Annotated[
        Literal["random", "grid"],
        Field(description="The hyper_tunning method to use during the job."),
    ] = "random"
    hyper_budget: Annotated[
        NonNegativeInt,
        Field(
            description='The budget to give for hyper tuning, only used if hyper_tuning is "random".'
        ),
    ] = 50
    # TODO: consider making a stratification /split submodel
    # TODO: change below to a boolean
    stratify_by_groups: Annotated[
        Literal["Y", "N"],
        Field(
            description="A field to indicate if the test/train dataset should be stratified by a group"
        ),
    ] = "N"
    groups: Annotated[
        str, Field(description="The name of the column to stratify the group by.")
    ] = None  # need to check
    # TODO consider making a sub-model for preprocessing
    standardize: Annotated[
        bool,
        Field(description="A bool to indicate if the data should be standardised."),
    ] = True
    balancing: Annotated[
        Literal["OVER", "UNDER", "NONE"],
        Field(
            description="A field to indicate which balancing methodology to use, only relevant for classification problems."
        ),
    ] = "NONE"
    fit_scorer: Annotated[
        Union[None, Literal[METRICS_NAMES_ALL]],
        Field(description="Which metric the models should optimis during training."),
    ] = None
    scorer_list: Annotated[
        Union[None, List[Literal[METRICS_NAMES_ALL]]],
        Field(description="Which metrics should be calculated for evaluation."),
    ] = []
    # TODO: consider adding a None option which will default to all applicable models.
    model_list: Annotated[
        List[Literal[MODEL_NAMES_ALL]],
        Field(description="A list of models to be trained in the job."),
    ]
    # TODO: check what the below actually drive
    encoding: Annotated[
        Literal["label", "onehot", None],
        Field(
            description="Which encoding method to use, only relevant in classification problems."
        ),
    ] = None
    autokeras_config: Annotated[
        Union[AutoKerasModel, None],
        Field(
            description="setting to be used for AutoKeras if it is chosen to be trained. Can be set to None if not selected."
        ),
    ] = AutoKerasModel()
    autolgbm_config: Annotated[
        Union[AutoLgbmModel, None],
        Field(
            description="settings to be used for AutoLgbm if chosen to be train. Can be set to None if not selected."
        ),
    ] = AutoLgbmModel()
    autoxgboost_config: Annotated[
        Union[AutoXgboostModel, None],
        Field(
            description="settings to be used for AutoXgboost if chosen to be train. Can be set to None if not selected."
        ),
    ] = AutoXgboostModel()
    feature_selection: Annotated[
        Union[FeatureSelectionModel, None],
        Field(
            description="Settings to be used for feature selection. If None no feature selection will be done."
        ),
    ] = FeatureSelectionModel()

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
