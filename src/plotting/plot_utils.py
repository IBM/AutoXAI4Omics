import matplotlib.pyplot as plt
from plotting.importance.perm_imp import permut_importance
from plotting.plots_both import (
    barplot_scorer,
    boxplot_scorer_cv,
    boxplot_scorer_cv_groupby,
)
from plotting.plots_clf import conf_matrix_plot, roc_curve_plot
from plotting.plots_reg import (
    correlation_plot,
    distribution_hist,
    histograms,
    joint_plot,
)
from plotting.shap.plots_shap import shap_force_plots, shap_plots

import logging
from utils.vars import CLASSIFICATION, REGRESSION

omicLogger = logging.getLogger("OmicLogger")


def define_plots(problem_type: str) -> dict[str:object]:
    """Define the plots for each problem type. This needs to be maintained manually.

    Returns
    -------
    problem_type : str
        a str that is either regression or classification

    Raises
    ------
    TypeError
        is raised if problem_type is not a str
    ValueError
        is raised if problem_type is not regresison or classification
    """

    if not isinstance(problem_type, str):
        raise TypeError(
            f"problem_type must be a str equal to {REGRESSION} or {CLASSIFICATION}"
        )
    if problem_type not in [REGRESSION, CLASSIFICATION]:
        raise ValueError(
            f"problem_type must be equal to {REGRESSION} or {CLASSIFICATION}"
        )

    omicLogger.debug("Define dict of plots...")
    # Some plots can only be done for a certain type of ML
    # Check here that the ones given are valid
    common_plot_dict = {
        "barplot_scorer": barplot_scorer,
        "boxplot_scorer": boxplot_scorer_cv,
        "boxplot_scorer_cv_groupby": boxplot_scorer_cv_groupby,
        "shap_plots": shap_plots,
        "shap_force_plots": shap_force_plots,
        "permut_imp_test": permut_importance,
    }

    if problem_type == CLASSIFICATION:
        plot_dict = {
            **common_plot_dict,
            "conf_matrix": conf_matrix_plot,
            "roc_curve": roc_curve_plot,
        }
    elif problem_type == REGRESSION:
        plot_dict = {
            **common_plot_dict,
            "corr": correlation_plot,
            "hist": distribution_hist,
            "hist_overlapped": histograms,
            "joint": joint_plot,
            "joint_dens": joint_plot,
        }
    return plot_dict


def create_fig(nrows=1, ncols=1, figsize=None):
    """
    Universal call to subplots to allow consistent specification of e.g. figsize
    """
    omicLogger.debug("Creating figure canvas...")
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, ax
