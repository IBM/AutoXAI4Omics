# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from ..prediction_model import PredictionModel as Model
import pytest
from copy import deepcopy

TEST_CONFIG = {
    "file_path": "data/50k_barley_row_type_processed.csv",
    "metadata_file": "data/50k_barley_row_type_processed_metadata.csv",
    "outfile_name": "testing_output",
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

    def test_filePathNulling(self):
        MODIFIED_CONFIG = deepcopy(TEST_CONFIG)
        MODIFIED_CONFIG["file_path"] = None

        model = Model(**MODIFIED_CONFIG)

        assert model.metadata_file is None
        assert model.outfile_name is None
