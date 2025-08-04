import pytest
import numpy as np
from ..standardisation import standardize_data
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import normaltest

np.random.seed(1234)
x = np.random.beta(10, 2, (2500, 10))
P_THRESHOLD = 0.01


class Test_standardize_data:
    def test_for_normalisation(self):
        _, pvals = normaltest(x)
        assert all(pvals < P_THRESHOLD)

        x_trans, SS = standardize_data(x)
        assert isinstance(SS, QuantileTransformer)

        _, pvals = normaltest(x_trans)
        assert all(pvals >= P_THRESHOLD)
