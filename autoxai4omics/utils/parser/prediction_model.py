# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Union
from pydantic import BaseModel, FilePath, model_validator, Field
from typing_extensions import Annotated


class PredictionModel(BaseModel):
    file_path: Annotated[
        Union[FilePath, None],
        Field(description="The file path to the data for predictions to be made in."),
    ]
    outfile_name: Annotated[
        str, Field(description="The name to give the file containing predictions.")
    ] = "prediction_results"
    metadata_file: Annotated[
        Union[FilePath, None],
        Field(
            description="The file path to the metadata associated with the datafile."
        ),
    ] = None

    @model_validator(mode="after")
    def check(self):
        if self.file_path is None:
            self.outfile_name = None
            self.metadata_file = None
        return self
