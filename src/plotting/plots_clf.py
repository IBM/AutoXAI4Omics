import matplotlib.cm as cmx
import numpy as np
import utils.load
from tensorflow.keras import backend as K
from utils.save import save_fig
from utils.utils import get_model_path

import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix, roc_curve
import time
from itertools import cycle
import logging

omicLogger = logging.getLogger("OmicLogger")


def roc_curve_plot(
    experiment_folder, model_list, x_test, y_test, save=True, holdout=False
):
    """
    Creates a ROC curve plot for each model. Saves them in separate files.
    """
    omicLogger.debug("Creating roc_curve_plot...")
    # Loop over the defined models
    for model_name in model_list:
        # Define the figure object
        fig, ax = plt.subplots()
        # Load the model
        model_path = get_model_path(experiment_folder, model_name)

        print(f"Plotting ROC Curve for {model_name}")
        model = utils.load.load_model(model_name, model_path)
        # Get the predictions
        y_pred = model.predict_proba(x_test)

        try:
            class_names = model.classes_.tolist()
        except AttributeError:
            print("Unable to get class names automatically")
            class_names = None

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(
                y_test, y_pred[:, i], pos_label=class_names[i]
            )
            roc_auc[i] = auc(fpr[i], tpr[i])

        ourcolors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for i, color in zip(range(len(class_names)), ourcolors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                label="ROC curve of class {0} (area = {1:0.2f})".format(
                    class_names[i], roc_auc[i]
                ),
            )

        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver operating characteristic - {model_name}")
        plt.legend(loc="lower right")

        # Save or show the plot
        if save:
            fname = f"{experiment_folder / 'graphs' / 'roc_curve'}_{model_name}"
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


def conf_matrix_plot(
    experiment_folder,
    model_list,
    x_test,
    y_test,
    normalize=False,
    save=True,
    holdout=False,
):
    """
    Creates a confusion matrix for each model. Saves them in separate files.
    """
    omicLogger.debug("Creating conf_matrix_plot...")
    # Loop over the defined models
    for model_name in model_list:
        # Define the figure object
        fig, ax = plt.subplots()
        # Load the model
        model_path = get_model_path(experiment_folder, model_name)

        print(f"Plotting Confusion Matrix for {model_name}")
        model = utils.load.load_model(model_name, model_path)
        # Get the predictions
        y_pred = model.predict(x_test)
        # Calc the confusion matrix
        print(y_pred)
        print(y_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        # Normalize the confusion matrix
        if normalize:
            conf_matrix = (
                conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
            )
        # Plot the confusion matrix
        im = ax.imshow(conf_matrix, interpolation="nearest", cmap=cmx.binary)
        ax.figure.colorbar(im, ax=ax)
        # Try to get the class names
        try:
            class_names = model.classes_.tolist()
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
        except AttributeError:
            print("Unable to get class names automatically")
            class_names = None
        # Setup the labels/ticks
        ax.set_xticks(np.arange(conf_matrix.shape[1]))
        ax.tick_params(axis="x", rotation=50)
        ax.set_yticks(np.arange(conf_matrix.shape[0]))
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        # Add the text annotations
        fmt = ".2f" if normalize else "d"
        # Threshold for black or white text
        thresh = conf_matrix.max() / 2.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                # Use white text if the colour is too dark
                ax.text(
                    j,
                    i,
                    format(conf_matrix[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black",
                )
        # Save or show the plot
        if save:
            fname = f"{experiment_folder / 'graphs' / 'conf_matrix'}_{model_name}"
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
