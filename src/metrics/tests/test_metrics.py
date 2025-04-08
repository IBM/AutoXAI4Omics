# Copyright 2024 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from ..metric_defs import METRICS
from ..metrics import define_scorers
from utils.vars import CLASSIFICATION, REGRESSION
from sklearn.metrics._scorer import _Scorer


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
        assert isinstance(sd[scorer], _Scorer)
