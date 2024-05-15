# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from pydantic import BaseModel, model_validator
from .data_model import DataModel
from .ml_model import MlModel
from .plotting_model import PlottingModel
from .tabular_model import TabularModel
from .microbiome_model import MicrobiomeModel
from .metabolomic_model import MetabolomicModel
from .geneExpression_model import GeneExpressionModel
from .prediction_model import PredictionModel


class ConfigModel(BaseModel):
    data: DataModel
    ml: MlModel
    plotting: PlottingModel = PlottingModel()
    tabular: TabularModel = TabularModel()
    microbiome: MicrobiomeModel = MicrobiomeModel()
    metabolomic: MetabolomicModel = MetabolomicModel()
    gene_expression: GeneExpressionModel = GeneExpressionModel(expression_type="OTHER")
    prediction: PredictionModel = None

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

        return self
