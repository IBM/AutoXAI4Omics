# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from ..vars import CLASSIFICATION, REGRESSION
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

        return self

    def validateWithProblemType(self, problemType):
        # TODO: trigger at higher levels
        if problemType not in [CLASSIFICATION, REGRESSION]:
            raise ValueError(
                f"problemType must be equal to either '{CLASSIFICATION}' or '{REGRESSION}"
            )

        CLF_SET = set(PLOTS_BOTH + PLOTS_CLF)
        REG_SET = set(PLOTS_BOTH + PLOTS_REG)
        if (problemType == CLASSIFICATION) and not (
            set(self.plot_method).issubset(CLF_SET)
        ):
            raise ValueError(
                f"These plots {','.join(set(self.plot_method)-CLF_SET)} are not valid for {problemType} problems"
            )

        if (problemType == REGRESSION) and not (
            set(self.plot_method).issubset(REG_SET)
        ):
            raise ValueError(
                f"These plots ({','.join(set(self.plot_method)-REG_SET)}) are not valid for {problemType} problems"
            )
