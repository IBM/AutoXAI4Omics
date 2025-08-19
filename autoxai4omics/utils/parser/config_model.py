# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from .data_model import DataModel
from .geneExpression_model import GeneExpressionModel
from .metabolomic_model import MetabolomicModel
from .microbiome_model import MicrobiomeModel
from .ml_model import MlModel
from .plotting_model import PlottingModel
from .prediction_model import PredictionModel
from .tabular_model import TabularModel
from pydantic import BaseModel, model_validator, Field
from typing import Union
from typing_extensions import Annotated


class ConfigModel(BaseModel):
    data: Annotated[
        DataModel,
        Field(
            description="A subsection corresponding to the data to be used in this job"
        ),
    ]
    ml: Annotated[
        MlModel,
        Field(
            description="A subsection corresponding to the machine learning settings to be used in this job"
        ),
    ]
    plotting: Annotated[
        PlottingModel,
        Field(
            description="A subsection corresponding to the plotting settings to be used in this job"
        ),
    ] = PlottingModel()
    tabular: Annotated[
        Union[TabularModel, None],
        Field(
            description="A subsection with settings if the data is of tabular type, this field can be None if not."
        ),
    ] = TabularModel()
    microbiome: Annotated[
        Union[MicrobiomeModel, None],
        Field(
            description="A subsection with settings if the data is of microbiome type, this field can be None if not."
        ),
    ] = MicrobiomeModel()
    metabolomic: Annotated[
        Union[MetabolomicModel, None],
        Field(
            description="A subsection with settings if the data is of metabolomic type, this field can be None if not."
        ),
    ] = MetabolomicModel()
    gene_expression: Annotated[
        Union[GeneExpressionModel, None],
        Field(
            description="A subsection with settings if the data is of gene expression type, this field can be None if not."
        ),
    ] = GeneExpressionModel(expression_type="OTHER")
    prediction: Annotated[
        Union[PredictionModel, None],
        Field(
            description="A subsection containing setting if a prediction job is to be run, this field can be None if not."
        ),
    ] = None

    @model_validator(mode="after")
    def check(self):

        self.plotting.validateWithProblemType(self.ml.problem_type)

        if self.data.data_type != "tabular":
            self.tabular = None
        if self.data.data_type != "gene_expression":
            self.gene_expression = None
        if self.data.data_type != "microbiome":
            self.microbiome = None
        if self.data.data_type != "metabolomic":
            self.metabolomic = None

        if self.data.data_type == "R2G":
            self.ml.standardize = False
            self.ml.feature_selection = None
            self.ml.balancing = "NONE"
            if self.prediction:
                self.prediction.metadata_file = None

        return self
