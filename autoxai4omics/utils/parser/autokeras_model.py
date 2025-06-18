# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Literal
from pydantic import BaseModel, confloat, PositiveInt

Dropout = confloat(strict=True, ge=0, le=1)


class AutoKerasModel(BaseModel):
    n_epochs: PositiveInt = 100
    batch_size: PositiveInt = 32
    verbose: bool = False
    n_blocks: PositiveInt = 3
    dropout: Dropout = 0.3  # type: ignore
    use_batchnorm: bool = True
    n_trials: PositiveInt = 4
    tuner: Literal["bayesian", "greedy", "hyperband", "random"] = "bayesian"
