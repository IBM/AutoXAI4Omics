# Copyright (c) 2025 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pytest
import numpy as np
from .. import feature_selection as fs
from sklearn.feature_selection import VarianceThreshold


np.random.seed(1234)

VAR = 0.5
SAMPLES = 30
FEATS = 10
FIXED = np.random.normal(size=(SAMPLES, FEATS))
SMALLER = np.random.normal(scale=(VAR**0.5), size=(SAMPLES, 1))
ZERO = np.random.normal(scale=0, size=(SAMPLES, 1))

X = np.concatenate((FIXED, SMALLER, ZERO), axis=1)


class Test_variance_removal:
    def test_zero_var_removal(self):
        x_trans, selector = fs.variance_removal(X)

        assert x_trans.shape == (SAMPLES, FEATS + 1)
        assert isinstance(selector, VarianceThreshold)

    def test_given_var_removal(self, given_var: float = VAR):
        x_trans, selector = fs.variance_removal(X, threshold=given_var)

        assert x_trans.shape == (SAMPLES, FEATS)
        assert isinstance(selector, VarianceThreshold)
