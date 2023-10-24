import pytest
import numpy as np
import pandas as pd
from ..data_split import std_split
from ...vars import REGRESSION, CLASSIFICATION


X_FULL = np.array(
    [
        [0.78394513, 0.28321302, 0.64717287],
        [0.0610643, 0.0978563, 0.18148352],
        [0.3370728, 0.72011669, 0.36164336],
        [0.63196984, 0.34719963, 0.62933416],
        [0.00310848, 0.8365884, 0.58731593],
        [0.93473098, 0.84219959, 0.30693875],
        [0.78715557, 0.59947952, 0.45340106],
        [0.967449, 0.46326428, 0.98278358],
        [0.28381946, 0.03484806, 0.64734372],
        [0.14845684, 0.87897958, 0.4762823],
    ]
)

Y_FULL = np.array([[0], [2], [0], [2], [1], [2], [0], [2], [1], [1]])


RESULTS = {
    REGRESSION: {
        "X_TRAIN": np.array(
            [
                [0.93473098, 0.84219959, 0.30693875],
                [0.967449, 0.46326428, 0.98278358],
                [0.0610643, 0.0978563, 0.18148352],
                [0.00310848, 0.8365884, 0.58731593],
                [0.3370728, 0.72011669, 0.36164336],
                [0.28381946, 0.03484806, 0.64734372],
            ]
        ),
        "Y_TRAIN": np.array([[2], [2], [2], [1], [0], [1]]),
        "X_TEST": np.array(
            [
                [0.63196984, 0.34719963, 0.62933416],
                [0.78394513, 0.28321302, 0.64717287],
                [0.14845684, 0.87897958, 0.4762823],
                [0.78715557, 0.59947952, 0.45340106],
            ]
        ),
        "Y_TEST": np.array([[2], [0], [1], [0]]),
    },
    CLASSIFICATION: {
        "X_TRAIN": np.array(
            [
                [0.93473098, 0.84219959, 0.30693875],
                [0.14845684, 0.87897958, 0.4762823],
                [0.78394513, 0.28321302, 0.64717287],
                [0.00310848, 0.8365884, 0.58731593],
                [0.3370728, 0.72011669, 0.36164336],
                [0.0610643, 0.0978563, 0.18148352],
            ]
        ),
        "Y_TRAIN": np.array([[2], [1], [0], [1], [0], [2]]),
        "X_TEST": np.array(
            [
                [0.967449, 0.46326428, 0.98278358],
                [0.78715557, 0.59947952, 0.45340106],
                [0.63196984, 0.34719963, 0.62933416],
                [0.28381946, 0.03484806, 0.64734372],
            ]
        ),
        "Y_TEST": np.array([[2], [0], [2], [1]]),
    },
}


class Test_std_split:
    def test_test_size_type(self):
        try:
            std_split(
                x_full=X_FULL,
                y_full=Y_FULL,
                test_size="2",
                seed_num=29292,
                problem_type=CLASSIFICATION,
            )
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_test_size_value(self):
        try:
            std_split(
                x_full=X_FULL,
                y_full=Y_FULL,
                test_size=2.0,
                seed_num=29292,
                problem_type=CLASSIFICATION,
            )
            assert False
        except ValueError:
            assert True
        except Exception:
            assert False

    def test_seed_type(self):
        try:
            std_split(
                x_full=X_FULL,
                y_full=Y_FULL,
                test_size=0.2,
                seed_num="29292",
                problem_type=CLASSIFICATION,
            )
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_problem_type_type(self):
        try:
            std_split(
                x_full=X_FULL,
                y_full=Y_FULL,
                test_size=0.2,
                seed_num=29292,
                problem_type=1,
            )
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_problem_type_value(self):
        try:
            std_split(
                x_full=X_FULL,
                y_full=Y_FULL,
                test_size=0.2,
                seed_num=29292,
                problem_type="WRONG",
            )
            assert False
        except ValueError:
            assert True
        except Exception:
            assert False

    def test_x_full_type(self):
        try:
            std_split(
                x_full=[],
                y_full=Y_FULL,
                test_size=0.2,
                seed_num=29292,
                problem_type=REGRESSION,
            )
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_y_train_type(self):
        try:
            std_split(
                x_full=X_FULL,
                y_full=[],
                test_size=0.2,
                seed_num=29292,
                problem_type=REGRESSION,
            )
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_shape_mismatch(self):
        try:
            std_split(
                x_full=X_FULL,
                y_full=Y_FULL[0:-1],
                test_size=0.2,
                seed_num=29292,
                problem_type=REGRESSION,
            )
            assert False
        except ValueError:
            assert True
        except Exception:
            assert False

    @pytest.mark.parametrize("problem_type", [CLASSIFICATION, REGRESSION])
    def test_outputs(self, problem_type):
        x_train, x_test, y_train, y_test = std_split(
            x_full=X_FULL,
            y_full=Y_FULL,
            test_size=0.4,
            seed_num=29292,
            problem_type=problem_type,
        )

        assert (x_train == RESULTS[problem_type]["X_TRAIN"]).all()
        assert (y_train == RESULTS[problem_type]["Y_TRAIN"]).all()
        assert (x_test == RESULTS[problem_type]["X_TEST"]).all()
        assert (y_test == RESULTS[problem_type]["Y_TEST"]).all()

    @pytest.mark.parametrize("problem_type", [CLASSIFICATION, REGRESSION])
    def test_outputs_df(self, problem_type):
        x_train, x_test, y_train, y_test = std_split(
            x_full=pd.DataFrame(X_FULL),
            y_full=pd.DataFrame(Y_FULL),
            test_size=0.4,
            seed_num=29292,
            problem_type=problem_type,
        )

        assert (x_train == RESULTS[problem_type]["X_TRAIN"]).all(axis=None)
        assert (y_train == np.reshape(RESULTS[problem_type]["Y_TRAIN"], (-1, 1))).all(axis=None)
        assert (x_test == RESULTS[problem_type]["X_TEST"]).all(axis=None)
        assert (y_test == np.reshape(RESULTS[problem_type]["Y_TEST"], (-1, 1))).all(axis=None)
