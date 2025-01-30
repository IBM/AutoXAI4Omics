# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from ..featureSelection_model import AutoModel, MethodModel, FeatureSelectionModel
from ...vars import CLASSIFICATION, REGRESSION
import pytest
from copy import deepcopy

TEST_CONFIG = {
    "k": "auto",
    "var_threshold": 0,
    "auto": {"min_features": 10, "interval": 1},
    "method": {"name": "SelectKBest", "metric": "f_classif"},
}
TEST_CONFIG_AUTO = {"min_features": 10, "interval": 1}
TEST_CONFIG_METHOD = {"name": "SelectKBest", "metric": "f_classif"}


class Test_AutoModel:
    def test_testConfig(self):
        try:
            AutoModel(**TEST_CONFIG_AUTO)
            assert True
        except Exception:
            assert False

    @pytest.mark.parametrize(
        "key", [k for k, v in AutoModel.model_fields.items() if v.is_required()]
    )
    def test_missing_required(self, key):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG_AUTO)
        del MODIFIED_CONFIG[key]

        try:
            AutoModel(**MODIFIED_CONFIG)
            assert False
        except Exception as e:
            errs = e.errors()
            assert len(errs) == 1
            errs = errs[0]
            assert errs["type"] == "missing"
            assert errs["loc"][0] == key

    def test_validation_problem_type(self):
        model = AutoModel(**TEST_CONFIG_AUTO)
        try:
            model.validateWithProblemType(None)
            assert False
        except ValueError as e:
            assert "problemType must be one of" in str(e)
        except Exception:
            assert False

    @pytest.mark.parametrize("problem_type", [CLASSIFICATION, REGRESSION])
    def test_validation_eval_defaults(self, problem_type):
        model = AutoModel(**TEST_CONFIG_AUTO)
        model.validateWithProblemType(problem_type)
        assert model.eval_model == (
            "RandomForestClassifier"
            if problem_type == CLASSIFICATION
            else "RandomForestRegressor"
        )

        assert model.eval_metric == (
            "f1_score" if problem_type == CLASSIFICATION else "mean_squared_error"
        )

    @pytest.mark.parametrize("problem_type", [CLASSIFICATION, REGRESSION])
    def test_validation_model(self, problem_type):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG_AUTO)
        MODIFIED_CONFIG["eval_model"] = (
            "RandomForestClassifier"
            if problem_type == REGRESSION
            else "RandomForestRegressor"
        )
        model = AutoModel(**MODIFIED_CONFIG)

        try:
            model.validateWithProblemType(problem_type)
            assert False
        except ValueError as e:
            assert "is not available for" in str(e)
        except Exception:
            assert False

    @pytest.mark.parametrize("problem_type", [CLASSIFICATION, REGRESSION])
    def test_validation_metric(self, problem_type):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG_AUTO)
        MODIFIED_CONFIG["eval_metric"] = (
            "f1_score" if problem_type == REGRESSION else "mean_squared_error"
        )
        model = AutoModel(**MODIFIED_CONFIG)

        try:
            model.validateWithProblemType(problem_type)
            assert False
        except ValueError as e:
            assert "is not available for" in str(e)
        except Exception:
            assert False


class Test_MethodModel:
    def test_testConfig(self):
        try:
            MethodModel(**TEST_CONFIG_METHOD)
            assert True
        except Exception:
            assert False

    @pytest.mark.parametrize(
        "key", [k for k, v in MethodModel.model_fields.items() if v.is_required()]
    )
    def test_missing_required(self, key):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG_METHOD)
        del MODIFIED_CONFIG[key]

        try:
            MethodModel(**MODIFIED_CONFIG)
            assert False
        except Exception as e:
            errs = e.errors()
            assert len(errs) == 1
            errs = errs[0]
            assert errs["type"] == "missing"
            assert errs["loc"][0] == key

    def test_validation_problem_type(self):
        model = MethodModel(**TEST_CONFIG_METHOD)
        try:
            model.validateWithProblemType(None)
            assert False
        except ValueError as e:
            assert "problemType must be one of" in str(e)
        except Exception:
            assert False

    @pytest.mark.parametrize("problem_type", [CLASSIFICATION, REGRESSION])
    def test_kbest_default(self, problem_type):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG_METHOD)
        MODIFIED_CONFIG["metric"] = None
        model = MethodModel(**MODIFIED_CONFIG)
        model.validateWithProblemType(problem_type)
        assert model.metric == (
            "f_classif" if problem_type == CLASSIFICATION else "f_regression"
        )

    @pytest.mark.parametrize("problem_type", [CLASSIFICATION, REGRESSION])
    def test_kbest_metric_incorrect(self, problem_type):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG_METHOD)
        MODIFIED_CONFIG["metric"] = (
            "f_classif" if problem_type == REGRESSION else "f_regression"
        )
        model = MethodModel(**MODIFIED_CONFIG)
        try:
            model.validateWithProblemType(problem_type)
            assert False
        except ValueError as e:
            assert "is not appropriate for a" in str(e)
        except Exception:
            assert False

    @pytest.mark.parametrize("problem_type", [CLASSIFICATION, REGRESSION])
    def test_rfe_default(self, problem_type):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG_METHOD)
        MODIFIED_CONFIG["name"] = "RFE"
        MODIFIED_CONFIG["metric"] = None
        model = MethodModel(**MODIFIED_CONFIG)
        model.validateWithProblemType(problem_type)
        assert model.estimator == (
            "RandomForestClassifier"
            if problem_type == CLASSIFICATION
            else "RandomForestRegressor"
        )

    @pytest.mark.parametrize("problem_type", [CLASSIFICATION, REGRESSION])
    def test_rfe_estimator_incorrect(self, problem_type):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG_METHOD)
        MODIFIED_CONFIG["name"] = "RFE"
        MODIFIED_CONFIG["metric"] = None
        MODIFIED_CONFIG["estimator"] = (
            "RandomForestClassifier"
            if problem_type == REGRESSION
            else "RandomForestRegressor"
        )
        model = MethodModel(**MODIFIED_CONFIG)
        try:
            model.validateWithProblemType(problem_type)
            assert False
        except ValueError as e:
            assert "is not appropriate for problem type" in str(e)
        except Exception:
            assert False


class Test_FeatureSelectionModel:
    def test_testConfig(self):
        try:
            FeatureSelectionModel(**TEST_CONFIG)
            assert True
        except Exception:
            assert False

    @pytest.mark.parametrize(
        "key", [k for k, v in MethodModel.model_fields.items() if v.is_required()]
    )
    def test_missing_required(self, key):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG)
        del MODIFIED_CONFIG[key]

        try:
            FeatureSelectionModel(**MODIFIED_CONFIG)
            assert False
        except Exception as e:
            errs = e.errors()
            assert len(errs) == 1
            errs = errs[0]
            assert errs["type"] == "missing"
            assert errs["loc"][0] == key

    def test_validation_problem_type(self):
        model = FeatureSelectionModel(**TEST_CONFIG)
        try:
            model.validateWithProblemType(None)
            assert False
        except ValueError as e:
            assert "problemType must be one of" in str(e)
        except Exception:
            assert False
