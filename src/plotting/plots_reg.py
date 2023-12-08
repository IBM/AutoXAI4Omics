import numpy as np
import scipy.stats as sp
import seaborn as sns
import utils.load
from tensorflow.keras import backend as K
from utils.save import save_fig
from utils.utils import get_model_path
import matplotlib.pyplot as plt
import time
import logging

omicLogger = logging.getLogger("OmicLogger")


def histograms(
    experiment_folder, model_list, x_test, y_test, class_name, save=True, holdout=False
):
    """
    Shows the histogram distribution of the true and predicted labels/values.

    Provides two figures side-by-side for better illustration.
    """
    omicLogger.debug("Creating histograms...")
    # Loop over the defined models
    for model_name in model_list:
        # Load the model
        model_path = get_model_path(experiment_folder, model_name)

        print(f"Plotting histogram for {model_name}")
        model = utils.load.load_model(model_name, model_path)
        # Get the predictions
        y_pred = model.predict(x_test)

        plt.hist([y_test, y_pred], label=["True", "Predicted"], alpha=0.5, bins=50)
        plt.legend(loc="upper right")
        axes = plt.gca()
        axes.set_ylim([0, 27])
        fig = plt.gcf()

        if save:
            fname = f"{experiment_folder / 'graphs' / 'hist_overlap'}_{model_name}"
            fname += "_holdout" if holdout else ""
            save_fig(fig, fname)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()

        # Clear keras and TF sessions/graphs etc.
        K.clear_session()


def correlation_plot(
    experiment_folder,
    model_list,
    x_test,
    y_test,
    class_name,
    fit_line=True,
    save=True,
    holdout=False,
):
    """
    Creates a correlation plot with a 1D line of best fit.
    """
    omicLogger.debug("Creating correlation_plot...")
    # Loop over the defined models
    for model_name in model_list:
        # Define the figure object
        fig, ax = plt.subplots()

        model_path = get_model_path(experiment_folder, model_name)

        print(f"Plotting Correlation Plot for {model_name}")
        model = utils.load.load_model(model_name, model_path)
        # Get the predictions
        y_pred = model.predict(x_test)
        # Calc the confusion matrix
        ax.scatter(y_test, y_pred, c="black", s=5, alpha=0.8)
        # Set the axis labels
        ax.set_xlabel("True Value")
        ax.set_ylabel("Predicted Value")
        # Set the title
        ax.set_title(f"Correlation for {class_name} using {model_name}")
        # Add a best fit line
        if fit_line:
            ax.plot(
                np.unique(y_test),
                np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)),
                linestyle="dashed",
                linewidth=2,
                color="dimgrey",
            )
        # Save or show the plot
        if save:
            fname = f"{experiment_folder / 'graphs' / 'corr_scatter'}_{model_name}"
            fname += "_holdout" if holdout else ""
            save_fig(fig, fname)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()
        # Clear keras and TF sessions/graphs etc.
        K.clear_session()


def distribution_hist(
    experiment_folder, model_list, x_test, y_test, class_name, save=True, holdout=False
):
    """
    Shows the histogram distribution of the true and predicted labels/values.

    Provides two figures side-by-side for better illustration.
    """
    omicLogger.debug("Creating distribution_hist...")
    # Loop over the defined models
    for model_name in model_list:
        # Define the figure object
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 10))

        # Load the model
        model_path = get_model_path(experiment_folder, model_name)

        print(f"Plotting histogram for {model_name}")
        model = utils.load.load_model(model_name, model_path)
        # Get the predictions
        y_pred = model.predict(x_test)
        # Left histograms
        ax_left.hist(y_test, bins=20, zorder=1, color="black", label="True")
        ax_left.hist(y_pred, bins=20, zorder=2, color="grey", label="Predicted")
        # Right histograms (exactly the same, just different zorder)
        ax_right.hist(y_test, bins=20, zorder=2, color="black", label="True")
        ax_right.hist(y_pred, bins=20, zorder=1, color="grey", label="Predicted")
        # Create a single legend
        handles, labels = ax_right.get_legend_handles_labels()
        # Add the legend
        fig.legend(handles, labels, loc="right")
        if save:
            fname = f"{experiment_folder / 'graphs' / 'hist'}_{model_name}"
            fname += "_holdout" if holdout else ""
            save_fig(fig, fname)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()
        # Clear keras and TF sessions/graphs etc.
        K.clear_session()


def joint_plot(
    experiment_folder,
    model_list,
    x_test,
    y_test,
    class_name,
    kind="reg",
    save=True,
    holdout=False,
):
    """
    Uses seaborn's jointplot to illustrate correlation and distribution.
    """
    omicLogger.debug("Creating joint_plot...")
    # Loop over the defined models
    for model_name in model_list:
        # Load the model
        model_path = get_model_path(experiment_folder, model_name)

        print(f"Plotting joint plot for {model_name}")
        model = utils.load.load_model(model_name, model_path)
        # Get the predictions
        y_pred = model.predict(x_test)
        sns.set(style="white")
        # Make the joint plot
        plot = sns.jointplot(
            x=y_test,
            y=y_pred,
            color="grey",
            kind=kind,
            # Space between the marginal and main axes
            space=0,
            # Control scatter properties
            scatter_kws={"s": 4},
            # Control line properties
            line_kws={"linewidth": 1.5, "color": "black", "linestyle": "dashed"},
            # Control histogram properties
            marginal_kws={"color": "midnightblue"},
        )

        # Set the labels
        plot.ax_joint.set_xlabel("True Value")
        plot.ax_joint.set_ylabel("Predicted Value")
        x0, x1 = plot.ax_joint.get_xlim()
        y0, y1 = plot.ax_joint.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        plot.ax_joint.plot(lims, lims, ":k")

        # Extract only the pearson val (ignore the p-value)
        def pearson(x, y):
            return sp.pearsonr(x, y)[0]

        # Add this value to the graph (warning: deprecated in the future)
        plot.annotate(
            pearson,
            loc="upper right",
            stat="Pearson's",
            borderpad=0.2,
            mode=None,
            edgecolor=None,
        )
        if save:
            if kind == "kde":
                fname = f"{experiment_folder / 'graphs' / 'joint_kde'}_{model_name}"
            else:
                fname = f"{experiment_folder / 'graphs' / 'joint'}_{model_name}"

            fname += "_holdout" if holdout else ""
            save_fig(plot, fname)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()
        # Clear keras and TF sessions/graphs etc.
        K.clear_session()
