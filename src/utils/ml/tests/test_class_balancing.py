# Copyright 2024 IBM Corp.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..class_balancing import oversample_data, undersample_data
import pytest
import numpy as np
import pandas as pd

X_DATA = np.array(
    [
        [0.5236519, 0.45115304, 0.34611641],
        [0.70496249, 0.12047246, 0.63791054],
        [0.39957773, 0.4111747, 0.66019212],
        [0.70502186, 0.08066132, 0.74075119],
        [0.3319641, 0.47226144, 0.14981086],
        [0.12167986, 0.26324856, 0.3780612],
        [0.43133041, 0.99980809, 0.92202931],
        [0.31347399, 0.7973591, 0.63000586],
        [0.69777093, 0.37433433, 0.86466214],
        [0.12166382, 0.97097703, 0.71749282],
    ]
)

Y_DATA = np.array([[1], [1], [0], [0], [0], [0], [0], [0], [1], [1]])

X_OS_EXP = np.array(
    [
        [0.5236519, 0.45115304, 0.34611641],
        [0.70496249, 0.12047246, 0.63791054],
        [0.39957773, 0.4111747, 0.66019212],
        [0.70502186, 0.08066132, 0.74075119],
        [0.3319641, 0.47226144, 0.14981086],
        [0.12167986, 0.26324856, 0.3780612],
        [0.43133041, 0.99980809, 0.92202931],
        [0.31347399, 0.7973591, 0.63000586],
        [0.69777093, 0.37433433, 0.86466214],
        [0.12166382, 0.97097703, 0.71749282],
        [0.5236519, 0.45115304, 0.34611641],
        [0.69777093, 0.37433433, 0.86466214],
    ]
)
Y_OS_EXP = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
OS_IND_EXP = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 8])

X_US_EXP = np.array(
    [
        [0.43133041, 0.99980809, 0.92202931],
        [0.12167986, 0.26324856, 0.3780612],
        [0.70502186, 0.08066132, 0.74075119],
        [0.31347399, 0.7973591, 0.63000586],
        [0.5236519, 0.45115304, 0.34611641],
        [0.70496249, 0.12047246, 0.63791054],
        [0.69777093, 0.37433433, 0.86466214],
        [0.12166382, 0.97097703, 0.71749282],
    ]
)
Y_US_EXP = np.array([0, 0, 0, 0, 1, 1, 1, 1])
US_IND_EXP = np.array([6, 5, 3, 7, 0, 1, 8, 9])


class Test_oversample:
    def test_seed_type(self):
        try:
            oversample_data(x_train=X_DATA, y_train=Y_DATA, seed="WRONG")
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_x_train_type(self):
        try:
            oversample_data(x_train=[], y_train=Y_DATA)
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_y_train_type(self):
        try:
            oversample_data(x_train=X_DATA, y_train=[])
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_shape_mismatch(self):
        try:
            oversample_data(x_train=X_DATA, y_train=Y_DATA[0:-1], seed=29292)
            assert False
        except ValueError:
            assert True
        except Exception:
            assert False

    def test_output(self):
        x_os, y_os, os_ind = oversample_data(x_train=X_DATA, y_train=Y_DATA, seed=29292)
        assert (x_os == X_OS_EXP).all(axis=None)
        assert (y_os == Y_OS_EXP).all(axis=None)
        assert (os_ind == OS_IND_EXP).all(axis=None)

    def test_output_df(self):
        x_os, y_os, os_ind = oversample_data(x_train=pd.DataFrame(X_DATA), y_train=pd.DataFrame(Y_DATA), seed=29292)
        assert (x_os == X_OS_EXP).all(axis=None)
        assert (y_os == np.reshape(Y_OS_EXP, (-1, 1))).all(axis=None)
        assert (os_ind == OS_IND_EXP).all(axis=None)


class Test_undersample:
    def test_seed_type(self):
        try:
            undersample_data(x_train=X_DATA, y_train=Y_DATA, seed="WRONG")
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_x_train_type(self):
        try:
            undersample_data(x_train=[], y_train=Y_DATA)
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_y_train_type(self):
        try:
            undersample_data(x_train=X_DATA, y_train=[])
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_shape_mismatch(self):
        try:
            undersample_data(x_train=X_DATA, y_train=Y_DATA[0:-1], seed=29292)
            assert False
        except ValueError:
            assert True
        except Exception:
            assert False

    def test_output(self):
        x_us, y_us, us_ind = undersample_data(x_train=X_DATA, y_train=Y_DATA, seed=29292)
        assert (x_us == X_US_EXP).all(axis=None)
        assert (y_us == Y_US_EXP).all(axis=None)
        assert (us_ind == US_IND_EXP).all(axis=None)

    def test_output_df(self):
        x_us, y_us, us_ind = undersample_data(x_train=pd.DataFrame(X_DATA), y_train=pd.DataFrame(Y_DATA), seed=29292)
        assert (x_us == X_US_EXP).all(axis=None)
        assert (y_us == np.reshape(Y_US_EXP, (-1, 1))).all(axis=None)
        assert (us_ind == US_IND_EXP).all(axis=None)
