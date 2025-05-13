# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from pydantic import BaseModel, PositiveInt


class AutoLgbmModel(BaseModel):
    verbose: bool = False
    n_trials: PositiveInt = 10
    timeout: PositiveInt = 1000
