# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from ..autolgbm_model import AutoLgbmModel as Model
import pytest
from copy import deepcopy

TEST_CONFIG = {
    "verbose": True,
    "n_trials": 4,
    "timeout": 1000,
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
