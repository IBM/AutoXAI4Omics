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

from ..plot_utils import define_plots, create_fig
import pytest
from utils.vars import CLASSIFICATION, REGRESSION


class Test_define_plots:
    def test_problem_type_type(self):
        try:
            define_plots(True)
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_problem_type_value(self):
        try:
            define_plots("WRONG")
            assert False
        except ValueError:
            assert True
        except Exception:
            assert False

    @pytest.mark.parametrize("problem_type", [CLASSIFICATION, REGRESSION])
    def test_check_keys(self, problem_type):
        plot_dict = define_plots(problem_type)

        common = {
            "barplot_scorer",
            "boxplot_scorer",
            "boxplot_scorer_cv_groupby",
            "shap_plots",
            "shap_force_plots",
            "permut_imp_test",
        }
        assert common.issubset(set(plot_dict.keys()))

        if problem_type == REGRESSION:
            reg_set = {
                "corr",
                "hist",
                "hist_overlapped",
                "joint",
                "joint_dens",
            }
            assert reg_set.issubset(set(plot_dict.keys()))
        elif problem_type == CLASSIFICATION:
            clf_set = {
                "conf_matrix",
                "roc_curve",
            }
            assert clf_set.issubset(set(plot_dict.keys()))


class Test_create_fig:
    def test_nrows_type(self):
        try:
            create_fig(nrows="False", ncols=1, figsize=(1, 2))
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_nrows_values(self):
        try:
            create_fig(nrows=-1, ncols=1, figsize=(1, 2))
            assert False
        except ValueError:
            assert True
        except Exception:
            assert False

    def test_ncols_type(self):
        try:
            create_fig(ncols="False", nrows=1, figsize=(1, 2))
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_ncols_values(self):
        try:
            create_fig(ncols=-1, nrows=1, figsize=(1, 2))
            assert False
        except ValueError:
            assert True
        except Exception:
            assert False

    def test_figsize_type(self):
        try:
            create_fig(ncols=1, nrows=1, figsize=bool)
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_figzise_tuple_type(self):
        try:
            create_fig(ncols=1, nrows=1, figsize=("WRONG", "WRONG"))
            assert False
        except TypeError:
            assert True
        except Exception:
            assert False

    def test_figzise_tuple_value(self):
        try:
            create_fig(ncols=1, nrows=1, figsize=(-1, 1))
            assert False
        except ValueError:
            assert True
        except Exception:
            assert False

    def test_figzise_tuple_value_quant(self):
        try:
            create_fig(ncols=1, nrows=1, figsize=(1,))
            assert False
        except ValueError:
            assert True
        except Exception:
            assert False

    def test_output(self):
        fig, ax = create_fig(ncols=2, nrows=2, figsize=(100, 100))

        assert ax.shape == (2, 2)
        assert all(fig.get_size_inches() == [100, 100])
