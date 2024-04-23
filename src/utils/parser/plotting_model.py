# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import List, Literal
from pydantic import (
    BaseModel,
    PositiveInt,
    model_validator,
)

PLOTS_BOTH = ["barplot_scorer", "boxplot_scorer", "shap_plots", "permut_imp_test"]
PLOTS_CLF = ["conf_matrix", "roc_curve"]
PLOTS_REG = ["hist_overlapped", "joint", "joint_dens", "corr"]

PLOTS_ALL = PLOTS_BOTH + PLOTS_CLF + PLOTS_REG


class PlottingModel(BaseModel):
    plot_method: List[Literal[tuple(PLOTS_ALL)]] = []
    top_feats_permImp: PositiveInt = 20
    top_feats_shap: PositiveInt = 20
    explanations_data: Literal["test", "exemplars", "all"] = "all"

    @model_validator(mode="after")
    def check(self):
        if "shap_plots" not in self.plot_method:
            self.top_feats_shap = None
            self.explanations_data = None

        if "permut_imp_test" not in self.plot_method:
            self.top_feats_permImp = None

        # TODO: validate classification/regression specific plots at higher levels
        return self
