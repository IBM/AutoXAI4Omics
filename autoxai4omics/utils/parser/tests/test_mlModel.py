# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from ..ml_model import MlModel as Model
from utils.vars import CLASSIFICATION, REGRESSION
import pytest
from copy import deepcopy

TEST_CONFIG = {
    CLASSIFICATION: {
        "seed_num": 42,
        "test_size": 0.2,
        "problem_type": "classification",
        "hyper_tuning": "random",
        "hyper_budget": 50,
        "stratify_by_groups": "N",
        "groups": "",
        "balancing": "NONE",
        "fit_scorer": "f1_score",
        "scorer_list": ["accuracy_score", "f1_score"],
        "model_list": [
            "RandomForestClassifier",
            "AdaBoostClassifier",
            "KNeighborsClassifier",
            "AutoXGBoost",
            "AutoLGBM",
            "AutoKeras",
        ],
        "encoding": None,
    },
    REGRESSION: {
        "seed_num": 42,
        "test_size": 0.2,
        "problem_type": "regression",
        "hyper_tuning": "random",
        "hyper_budget": 50,
        "stratify_by_groups": "N",
        "groups": "",
        "balancing": "NONE",
        "fit_scorer": "mean_absolute_percentage_error",
        "scorer_list": ["mean_squared_error", "mean_absolute_error", "r2_score"],
        "model_list": [
            "RandomForestRegressor",
            "AdaBoostRegressor",
            "KNeighborsRegressor",
            "AutoXGBoost",
            "AutoLGBM",
            "AutoKeras",
        ],
        "encoding": None,
    },
}


@pytest.mark.parametrize("problem_type", [CLASSIFICATION, REGRESSION])
class Test_Model:

    def test_testConfig(self, problem_type):
        try:
            Model(**TEST_CONFIG[problem_type])
            assert True
        except Exception:
            assert False

    @pytest.mark.parametrize(
        "key", [k for k, v in Model.model_fields.items() if v.is_required()]
    )
    def test_missing_required(self, key, problem_type):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG[problem_type])
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

    def test_hyper_budget_setting(self, problem_type):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG[problem_type])
        MODIFIED_CONFIG["hyper_tuning"] = "grid"
        model = Model(**MODIFIED_CONFIG)
        assert model.hyper_budget is None

    def test_default_fit_scored(self, problem_type):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG[problem_type])
        MODIFIED_CONFIG["fit_scorer"] = None
        model = Model(**MODIFIED_CONFIG)
        assert model.fit_scorer == (
            "f1_score"
            if problem_type == CLASSIFICATION
            else "mean_absolute_percentage_error"
        )

    def test_invalid_fit_scored(self, problem_type):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG[problem_type])
        MODIFIED_CONFIG["fit_scorer"] = (
            "f1_score"
            if problem_type == REGRESSION
            else "mean_absolute_percentage_error"
        )
        try:
            Model(**MODIFIED_CONFIG)
            assert False
        except ValueError as e:
            assert "fit_scorer must be one of" in str(e)
        except Exception:
            assert False

    def test_default_scorer_list(self, problem_type):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG[problem_type])
        MODIFIED_CONFIG["scorer_list"] = None
        model = Model(**MODIFIED_CONFIG)
        assert model.scorer_list == [
            (
                "f1_score"
                if problem_type == CLASSIFICATION
                else "mean_absolute_percentage_error"
            )
        ]

        MODIFIED_CONFIG["scorer_list"] = []
        model = Model(**MODIFIED_CONFIG)
        assert model.scorer_list == [
            (
                "f1_score"
                if problem_type == CLASSIFICATION
                else "mean_absolute_percentage_error"
            )
        ]

    def test_invalid_scorer_list(self, problem_type):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG[problem_type])
        MODIFIED_CONFIG["scorer_list"] = (
            TEST_CONFIG[CLASSIFICATION]["scorer_list"]
            if problem_type == REGRESSION
            else TEST_CONFIG[REGRESSION]["scorer_list"]
        )
        try:
            Model(**MODIFIED_CONFIG)
            assert False
        except ValueError as e:
            assert "Non-valid options for scorer_list" in str(e)
        except Exception:
            assert False

    def test_invalid_model_list(self, problem_type):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG[problem_type])
        MODIFIED_CONFIG["model_list"] = (
            TEST_CONFIG[CLASSIFICATION]["model_list"]
            if problem_type == REGRESSION
            else TEST_CONFIG[REGRESSION]["model_list"]
        )
        try:
            Model(**MODIFIED_CONFIG)
            assert False
        except ValueError as e:
            assert "Non-valid options for model_list" in str(e)
        except Exception:
            assert False

    def test_nulling_encoding(self, problem_type):
        if problem_type == CLASSIFICATION:
            pytest.skip()

        MODIFIED_CONFIG = deepcopy(TEST_CONFIG[problem_type])
        MODIFIED_CONFIG["encoding"] = "onehot"
        model = Model(**MODIFIED_CONFIG)

        assert model.encoding is None

    @pytest.mark.parametrize("autoModel", ["AutoKeras", "AutoLGBM", "AutoXGBoost"])
    def test_nulling_auto_config(self, problem_type, autoModel):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG[problem_type])
        MODIFIED_CONFIG["model_list"].remove(autoModel)
        model = Model(**MODIFIED_CONFIG)

        assert model.model_dump()[f"{autoModel.lower()}_config"] is None
