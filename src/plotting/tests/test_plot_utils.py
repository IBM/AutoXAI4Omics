from ..plot_utils import define_plots
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
