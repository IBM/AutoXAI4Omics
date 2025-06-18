# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from ..microbiome_model import MicrobiomeModel as Model
import pytest
from copy import deepcopy

TEST_CONFIG = {
    "collapse_tax": "genus",
    "min_reads": 1,
    "norm_reads": 1,
    "filter_abundance": 1,
    "filter_prevalence": 0.5,
    "filter_microbiome_samples": None,
    "remove_classes": None,
    "merge_classes": None,
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

    def test_prevalence_low(self):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG)
        MODIFIED_CONFIG["filter_prevalence"] = -0.5
        try:
            Model(**MODIFIED_CONFIG)
            assert False
        except Exception as e:
            errs = e.errors()
            assert len(errs) == 1
            errs = errs[0]
            assert errs["type"] == "greater_than_equal"

    def test_prevalence_high(self):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG)
        MODIFIED_CONFIG["filter_prevalence"] = 5
        try:
            Model(**MODIFIED_CONFIG)
            assert False
        except Exception as e:
            errs = e.errors()
            assert len(errs) == 1
            errs = errs[0]
            assert errs["type"] == "less_than_equal"

    # TODO: more tests if model changes
