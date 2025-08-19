# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Literal, Union
from pydantic import BaseModel, FilePath, DirectoryPath, model_validator, Field
from typing_extensions import Annotated


class DataModel(BaseModel):
    name: Annotated[str, Field(description="The name that is to be given to this job.")]
    file_path: Annotated[
        FilePath,
        Field(description="The path to the data that this job is to be run on."),
    ]
    metadata_file: Annotated[
        Union[FilePath, None],
        Field(
            description="The metadata file that accompanies the data to be trained on."
        ),
    ] = None
    file_path_holdout_data: Annotated[
        Union[FilePath, None],
        Field(
            description="The path to the dataset that is to be used as a holdout set."
        ),
    ] = None
    metadata_file_holdout_data: Annotated[
        Union[FilePath, None],
        Field(
            description="The path to the metadata file that accompanies the holdout set"
        ),
    ] = None
    save_path: Annotated[
        DirectoryPath, Field(description="The path where the results shall be save.")
    ] = "/experiments/"
    target: Annotated[
        str,
        Field(
            description="The name of the column in the dataset or in the metadata that is to be predicted."
        ),
    ]
    data_type: Annotated[
        Literal[
            "tabular", "gene_expression", "microbiome", "metabolomic", "other", "R2G"
        ],
        Field(
            description='The type of the data that this job will be run on. Note - "R2G" means Ready to Go, meaning that no preprocessing is required and that the dataset is already split into train/test sets (denoted by a column called "set") and has labels present in a "label" column.'
        ),
    ]

    @model_validator(mode="after")
    def check(self):
        if self.data_type == "R2G":
            self.metadata_file = None
            self.metadata_file_holdout_data = None
        return self
