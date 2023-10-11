import json
import joblib

import pandas as pd

from mode_plotting import omicLogger


def save_config(experiment_folder, config_path, config_dict):
    """
    Save the config into the results folder for easy access (storage is cheap right?)
    """
    # Construct the file name
    fname = experiment_folder / config_path.name
    with open(fname, "w") as outfile:
        json.dump(config_dict, outfile, indent=4)


def save_exemplars_SHAP_values(
    config_dict,
    experiment_folder,
    feature_names,
    model_name,
    class_names,
    exemplars_selected,
    fold_id,
):
    # Deal with classification differently, classification has shap values for each class
    # Get the SHAP values (global impact) sorted from the highest to the lower (absolute value)
    if config_dict["ml"]["problem_type"] == "classification":
        # XGBoost for binary classification seems to return the SHAP values only for class 1
        if model_name == "xgboost" and len(class_names) == 2:
            df_exemplars = pd.DataFrame(data=exemplars_selected, columns=feature_names)
            fname_exemplars = f"{experiment_folder / 'results' / 'exemplars_SHAP_values'}_{model_name}_{fold_id}"
            df_exemplars.to_csv(fname_exemplars + ".txt")

        # When class > 2 (or class > 1 for all the models except XGBoost) SHAP return a list of SHAP value matrices. One for each class.
        else:
            print(type(exemplars_selected))
            print(len(exemplars_selected))
            print(len(exemplars_selected) == len(class_names))

            for i in range(len(exemplars_selected)):
                print("Class: " + str(i))
                print("Class name: " + str(class_names[i]))
                df_exemplars = pd.DataFrame(data=exemplars_selected[i], columns=feature_names)
                fname_exemplars = f"{experiment_folder / 'results' / 'exemplars_SHAP_values'}_{model_name}_{class_names[i]}_{i}_{fold_id}"
                df_exemplars.to_csv(fname_exemplars + ".txt")

    # Deal with regression
    else:
        df_exemplars = pd.DataFrame(data=exemplars_selected, columns=feature_names)
        fname_exemplars = f"{experiment_folder / 'results' / 'exemplars_SHAP_values'}_{model_name}_{fold_id}"
        df_exemplars.to_csv(fname_exemplars + ".txt")


def save_explainer(experiment_folder, model_name, explainer):
    save_name = f"{experiment_folder / 'models' / 'explainers' / 'shap'}_{model_name}.pkl"
    with open(save_name, "wb") as f:
        joblib.dump(explainer, f)


def save_fig(fig, fname, dpi=200, fig_format="png"):
    omicLogger.debug(f"Saving figure ({fname})to file...")
    print(f"Save location: {fname}.{fig_format}")
    fig.savefig(
        f"{fname}.{fig_format}",
        dpi=dpi,
        format=fig_format,
        bbox_inches="tight",
        transparent=False,
    )
