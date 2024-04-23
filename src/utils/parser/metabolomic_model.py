# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Union
from pydantic import (
    BaseModel,
    NonNegativeFloat,
    NonNegativeInt,
    conlist,
    NewPath,
    model_validator,
)


MeasurementFilter = conlist(item_type=NonNegativeInt, min_length=2, max_length=2)


class MetabolomicModel(BaseModel):
    filter_metabolomic_sample: NonNegativeFloat = 0
    filter_measurements: MeasurementFilter = [0, 0]
    output_file_met: Union[NewPath, None] = None
    output_metadata: Union[NewPath, None] = None

    @model_validator(mode="after")
    def check(self):
        if self.output_file_met is None and self.output_metadata is not None:
            self.output_metadata = None
        return self
