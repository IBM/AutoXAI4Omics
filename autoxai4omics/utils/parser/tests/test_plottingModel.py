# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from ..plotting_model import PlottingModel as Model
from ..plotting_model import PLOTS_ALL
from ...vars import CLASSIFICATION, REGRESSION
import pytest
from copy import deepcopy

TEST_CONFIG = {
    "plot_method": PLOTS_ALL,
    "top_feats_permImp": 20,
    "top_feats_shap": 20,
    "explanations_data": "all",
}


class Test_Model:
    def test_testConfig(self):
        try:
            Model(**TEST_CONFIG)
            assert True
        except Exception:
            assert False

    @pytest.mark.parametrize(
        "key", [k for k, v in Model.model_fields.items() if v.is_required()]
    )
    def test_missing_required(self, key):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG)
        del MODIFIED_CONFIG[key]

        try:
            Model(**MODIFIED_CONFIG)
            assert False
        except Exception as e:
            errs = e.errors()
            assert len(errs) == 1
            errs = errs[0]
            assert errs["type"] == "missing"
            assert errs["loc"][0] == key

    def test_shapNulling(self):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG)
        MODIFIED_CONFIG["plot_method"].remove("shap_plots")

        model = Model(**MODIFIED_CONFIG)

        assert model.top_feats_shap is None
        assert model.explanations_data is None

    def test_permuteNulling(self):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG)
        MODIFIED_CONFIG["plot_method"].remove("permut_imp_test")

        model = Model(**MODIFIED_CONFIG)

        assert model.top_feats_permImp is None

    def test_validation_problem_type(self):
        model = Model(**TEST_CONFIG)
        try:
            model.validateWithProblemType(None)
            assert False
        except ValueError as e:
            assert "problemType must be equal to either" in str(e)
        except Exception:
            assert False

    @pytest.mark.parametrize("problem", [CLASSIFICATION, REGRESSION])
    def test_validateWithProblemType(self, problem):
        try:
            model = Model(**TEST_CONFIG)
            model.validateWithProblemType(problemType=problem)
            assert False
        except ValueError as e:
            assert f"are not valid for {problem} problems" in str(e)
        except Exception:
            assert False
