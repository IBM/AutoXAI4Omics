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


from models.custom_model import CustomModel
from tensorflow.keras import backend as K
from utils.load import load_model
from utils.save import save_fig
from utils.utils import get_model_path, pretty_names
from utils.vars import CLASSIFICATION
import eli5
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

omicLogger = logging.getLogger("OmicLogger")


def permut_importance(
    experiment_folder,
    seed_num: int,
    model_list: list[str],
    fit_scorer: str,
    problem_type: str,
    scorer_dict,
    feature_names,
    data,
    labels,
    num_features,
    cv=None,
    save=True,
    holdout=False,
):
    """
    Use ELI5's permutation importance to assess the importance of the features.

    Note that in scikit-learn 0.21 there should be a version of this in the new model inspection module.
    This may be useful to use/watch for the future.
    """
    omicLogger.debug("Creating permut_importance...")
    print(feature_names)
    print(type(feature_names))

    # Loop over the defined models
    for model_name in model_list:
        if model_name == "mlp_ens":
            continue
        # Define the figure object
        fig, ax = plt.subplots()
        # Load the model
        model_path = get_model_path(experiment_folder, model_name)
        print(f"Plotting permutation importance for {model_name}")

        print("Model path")
        print(model_path)
        print("Model name")
        print(model_name)

        model = load_model(model_name, model_path)
        # Select the scoring function
        scorer_func = scorer_dict[fit_scorer]
        # Handle the custom model
        if isinstance(model, tuple(CustomModel.__subclasses__())):
            # Remove the test data to avoid any saving
            if model.data_test is not None:
                model.data_test = None
            if model.labels_test is not None:
                model.labels_test = None

        importances = eli5.sklearn.PermutationImportance(
            model,
            scoring=scorer_func,
            random_state=seed_num,
            cv=cv,
        ).fit(data, labels)

        a = np.asarray(importances.results_)

        # Get the top x indices of the features
        top_indices = np.argsort(np.median(a, axis=0))[::-1][:num_features]

        # Get the names of these features
        if isinstance(feature_names, list):
            top_features = np.array(feature_names)[top_indices]
        else:
            top_features = feature_names.values[top_indices]

        # Get the top values
        top_values = a[:, top_indices]
        # Split the array up for the boxplot func
        top_values = [top_values[:, i] for i in range(top_values.shape[1])]

        top_feature_info = {
            "Features_names": top_features,
            "Features_importance_value": top_values,
        }

        df_topfeature_info = pd.DataFrame(
            top_feature_info, columns=["Features_names", "Features_importance_value"]
        )

        if cv == "prefit":
            df_topfeature_info.to_csv(
                f"{experiment_folder / 'results' / 'permutimp_TopFeatures_info'}_{model_name}"
                + ".csv"
            )
        else:
            df_topfeature_info.to_csv(
                f"{experiment_folder / 'results' / 'permutimp_TopFeatures_info'}_{model_name}_cv-{cv}"
                + ".csv"
            )

        # Make a horizontal boxplot ordered by the magnitude
        ax = sns.boxplot(x=top_values, y=top_features, orient="h", ax=ax)
        if problem_type == CLASSIFICATION:
            ax.set_xlabel(f"{pretty_names(fit_scorer, 'score')} Decrease")
        else:
            ax.set_xlabel(f"{pretty_names(fit_scorer, 'score')} Increase")
            ax.set_ylabel("Features")

        # Save the plot
        if save:
            fname = f"{experiment_folder / 'graphs' / 'permutimp'}_{model_name}"
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
