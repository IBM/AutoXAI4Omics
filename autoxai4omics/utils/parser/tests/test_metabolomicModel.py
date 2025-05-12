# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from ..metabolomic_model import MetabolomicModel as Model
import pytest
from copy import deepcopy

TEST_CONFIG = {
    "filter_metabolomic_sample": 0,
    "filter_measurements": [0, 0],
    "output_file_met": "data/TEST.csv",
    "output_metadata": "data/TEST.csv",
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

    def test_measurementsFieldLength_long(self):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG)
        MODIFIED_CONFIG["filter_measurements"] = [0, 0, 0]
        try:
            Model(**MODIFIED_CONFIG)
            assert False
        except Exception as e:
            errs = e.errors()
            assert len(errs) == 1
            errs = errs[0]
            assert errs["type"] == "too_long"

    def test_measurementsFieldLength_short(self):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG)
        MODIFIED_CONFIG["filter_measurements"] = [0]
        try:
            Model(**MODIFIED_CONFIG)
            assert False
        except Exception as e:
            errs = e.errors()
            assert len(errs) == 1
            errs = errs[0]
            assert errs["type"] == "too_short"

    def test_outputNulling(self):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG)
        MODIFIED_CONFIG["output_file_met"] = None

        model = Model(**MODIFIED_CONFIG)
        assert model.output_metadata is None
