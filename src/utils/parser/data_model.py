# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Literal, Union
from pydantic import BaseModel, FilePath, DirectoryPath, model_validator


class DataModel(BaseModel):
    name: str
    file_path: FilePath
    metadata_file: Union[FilePath, None] = None
    file_path_holdout_data: Union[FilePath, None] = None
    metadata_file_holdout_data: Union[FilePath, None] = None
    save_path: DirectoryPath = "/experiments/"
    target: str
    data_type: Literal[
        "tabular", "gene_expression", "microbiome", "metabolomic", "other", "R2G"
    ]

    @model_validator(mode="after")
    def check(self):
        if self.data_type == "R2G":
            self.metadata_file = None
            self.metadata_file_holdout_data = None
        return self
