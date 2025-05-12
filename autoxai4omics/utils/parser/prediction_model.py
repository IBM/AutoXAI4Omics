# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Union
from pydantic import (
    BaseModel,
    FilePath,
    model_validator,
)


class PredictionModel(BaseModel):
    file_path: Union[FilePath, None]
    outfile_name: str = "prediction_results"
    metadata_file: Union[FilePath, None] = None

    @model_validator(mode="after")
    def check(self):
        if self.file_path is None:
            self.outfile_name = None
            self.metadata_file = None
        return self
