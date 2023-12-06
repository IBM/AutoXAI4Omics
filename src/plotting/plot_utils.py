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


def create_fig(nrows: int = 1, ncols: int = 1, figsize: tuple[int] = None):
    """Universal call to subplots to allow consistent specification of e.g. figsize

    Parameters
    ----------
    nrows : int, optional
        the number of plot rows to have in the plot, by default 1
    ncols : int, optional
        the numbers of plot columns to have in the plot, by default 1
    figsize : tuple[int], optional
        the dimensions to have the figure size, by default None

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    TypeError
        is raised if nrows or ncols is not an int
    ValueError
        is raised if nrows or ncols is not an greater than 0
    TypeError
        is raised if figise is not None or a tuple of ints
    ValueError
        is raised if figsize does not contain 2 int or if any of the ints are less than 0
    """
    if not isinstance(nrows, int):
        raise TypeError("nrows must be an int greater than 0")
    elif nrows < 1:
        raise ValueError("nrows must be greater that 0")

    if not isinstance(ncols, int):
        raise TypeError("ncols must be an int greater than 0")
    elif ncols < 1:
        raise ValueError("ncols must be greater that 0")

    if not (isinstance(figsize, tuple) or figsize is None):
        raise TypeError(
            f"figsize must be a tuple of ints or None. recieved {type(figsize)}"
        )
    if isinstance(figsize, tuple):
        if not all([isinstance(x, int) for x in figsize]):
            raise TypeError("figsize must onyl contain ints")
        elif len(figsize) != 2:
            raise ValueError(
                f"figsize must contain 2 elements, only recieved {len(figsize)}"
            )
        elif not all([x > 0 for x in figsize]):
            raise ValueError("Elements of figz size must be ints greater than 0")

    omicLogger.debug("Creating figure canvas...")
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, ax
