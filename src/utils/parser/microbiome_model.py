# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Union, Literal, List, Dict
from pydantic import (
    BaseModel,
    PositiveInt,
    confloat,
)

Prevalence = confloat(strict=True, le=1, ge=0)


class MicrobiomeModel(BaseModel):
    collapse_tax: Literal["genus", "species", None] = None
    min_reads: Union[PositiveInt, None] = None
    norm_reads: Union[PositiveInt, None] = None
    filter_abundance: Union[PositiveInt, None] = None
    filter_prevalence: Union[Prevalence, None] = None
    filter_microbiome_samples: Union[dict, None] = None
    remove_classes: Union[List[str], None] = None
    merge_classes: Union[None, Dict[str, List[str]]] = None

    # TODO: check if conditional validation is needed
