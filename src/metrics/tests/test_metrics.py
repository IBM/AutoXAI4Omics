import pytest
from ..metric_defs import METRICS
from ..metrics import define_scorers
from utils.vars import CLASSIFICATION, REGRESSION
from sklearn.metrics._scorer import _PredictScorer, _ProbaScorer


class Test_define_scorers:
    def test_problem_type_type(self):
        try:
            define_scorers(problem_type=1, scorer_list=[])
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_problem_type_value(self):
        try:
            define_scorers(problem_type="1", scorer_list=[])
            assert False
        except ValueError:
            assert True
        except Exception:
            assert False

    def test_scorer_list_type(self):
        try:
            define_scorers(problem_type=CLASSIFICATION, scorer_list=1)
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_scorer_list_list_empty(self):
        try:
            define_scorers(problem_type=CLASSIFICATION, scorer_list=[])
            assert False
        except ValueError:
            assert True
        except Exception:
            assert False

    def test_scorer_list_list_type(self):
        try:
            define_scorers(problem_type=CLASSIFICATION, scorer_list=[1])
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    @pytest.mark.parametrize("problem_type", [CLASSIFICATION, REGRESSION])
    def test_scorer_list_value(self, problem_type):
        try:
            define_scorers(problem_type=problem_type, scorer_list=["FAKE"])
            assert False
        except ValueError:
            assert True
        except Exception:
            assert False

    @pytest.mark.parametrize("problem_type", [CLASSIFICATION, REGRESSION])
    def test_outputs(self, problem_type):
        scorer = list(METRICS[problem_type].keys())[0]

        sd = define_scorers(problem_type=problem_type, scorer_list=[scorer])

        assert len(sd) == 1
        assert list(sd.keys())[0] == scorer
        assert isinstance(sd[scorer], (_PredictScorer, _ProbaScorer))
