from tensorflow.keras import backend as K
import utils.load
from utils.utils import pretty_names
from utils.save import save_fig
from utils.utils import get_model_path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from utils.vars import CLASSIFICATION, REGRESSION
import time
import logging

omicLogger = logging.getLogger("OmicLogger")


def select_explainer(model, model_name, df_train, problem_type):
    """
    Select the appropriate SHAP explainer for each model
    """
    # Select the right explainer
    # Note that, for a multi-class (non-binary) problem gradboost cannot use the TreeExplainer
    if model_name in ["xgboost", "rf"]:
        explainer = shap.TreeExplainer(model)
    elif model_name in ["mlp_keras"]:
        explainer = shap.DeepExplainer(model.model, df_train.values)
    elif model_name in ["AutoLGBM", "AutoXGBoost"]:
        explainer = shap.TreeExplainer(model.model.model)
    else:
        # KernelExplainer can be very slow, so use their KMeans to speed it up
        # Results are approximate
        df_train_km = shap.kmeans(df_train, 5)
        # For classification we use the predict_proba
        if problem_type == CLASSIFICATION:
            explainer = shap.KernelExplainer(model.predict_proba, df_train_km)
        # Otherwise just use predict
        elif problem_type == REGRESSION:
            explainer = shap.KernelExplainer(model.predict, df_train_km)
    return explainer


def shap_force_plots(
    experiment_folder,
    model_list,
    problem_type,
    x_test,
    y_test,
    feature_names,
    x,
    y,
    x_train,
    data_forexplanations,
    class_col="?",
    top_exemplars=0.1,
    save=True,
    holdout=False,
):
    """
    Wrapper to create a SHAP force plot for the top exemplar of each class for each model.
    """
    omicLogger.debug("Creating shap_force_plots...")
    # Convert the data into dataframes to ensure features are displayed
    if data_forexplanations == "all":
        data = x
        y_data = y
    elif data_forexplanations == "test":
        data = x_test
        y_data = y_test

    # Convert the data into dataframes to ensure features are displayed
    pd.DataFrame(data=data, columns=feature_names)
    df_train = pd.DataFrame(data=x_train, columns=feature_names)

    # Get the model paths
    for model_name in model_list:
        model_path = get_model_path(experiment_folder, model_name)

        print(f"Plotting SHAP for {model_name}")
        model = utils.load.load_model(model_name, model_path)

        # Select the right explainer from SHAP
        explainer = select_explainer(model, model_name, df_train, problem_type)
        shap_values = explainer.shap_values(data)

        # Handle classification and regression differently
        if problem_type == CLASSIFICATION:
            # Try to get the class names
            shap_force_clf(
                experiment_folder,
                feature_names,
                save,
                holdout,
                data,
                y_data,
                model_name,
                model,
                explainer,
                shap_values,
            )

        # Different exemplar calc for regression
        elif problem_type == REGRESSION:
            # Containers to avoid repetition with calling graph func
            shap_force_reg(
                experiment_folder,
                feature_names,
                class_col,
                top_exemplars,
                save,
                holdout,
                data,
                y_data,
                model_name,
                model,
                explainer,
                shap_values,
            )
            # Clear everything
        # Clear keras and TF sessions/graphs etc.
        K.clear_session()


def shap_force_reg(
    experiment_folder,
    feature_names,
    class_col,
    top_exemplars,
    save,
    holdout,
    data,
    y_data,
    model_name,
    model,
    explainer,
    shap_values,
):
    names = []
    exemplar_indices = []
    # Get the predictions
    preds = model.predict(data).flatten()
    # Calculate the difference in predictions
    dists = np.abs(y_data - preds)
    # Select the max and min top exemplars (i.e. closest to the max and min values for the target)
    if top_exemplars is not None:
        indices = dists.argsort()
        # Select the top percentage to choose from
        num_indices = int(len(preds) * top_exemplars)
        # Select the top exemplars
        exemplar_index = indices[:num_indices]
        # With clashes, it takes the first found (which is good as this corresponds to the lower prediction
        # error)
        top_min_index = exemplar_index[np.argmin(y_data[exemplar_index])]
        exemplar_indices.append(top_min_index)
        names.append("min")
        top_max_index = exemplar_index[np.argmax(y_data[exemplar_index])]
        exemplar_indices.append(top_max_index)
        names.append("max")
        # Otherwise we just take our single best prediction
    else:
        exemplar_indices.append(dists.argmin())
        names.append("closest")
        # Create a plot for each of the selected exemplars
    for name, exemplar_index in zip(names, exemplar_indices):
        # Create the plot
        fig = shap.force_plot(
            explainer.expected_value,
            shap_values[exemplar_index],
            data[exemplar_index],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            text_rotation=30,
        )
        # Setup the title
        fig.suptitle(
            f"SHAP Force Plot for top exemplar using {pretty_names(model_name, 'model')} for {class_col}"
            + f"({name})",
            fontsize=16,
            y=1.4,
        )
        # Save the plot
        if save:
            fname = f"{experiment_folder / 'graphs' / 'shap_force_single'}_{model_name}_{name}"
            fname += "_holdout" if holdout else ""
            save_fig(fig, fname)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()


def shap_force_clf(
    experiment_folder,
    feature_names,
    save,
    holdout,
    data,
    y_data,
    model_name,
    model,
    explainer,
    shap_values,
):
    try:
        class_names = model.classes_.tolist()
    except AttributeError:
        print("Unable to get class names automatically - classes will be encoded")
        # Hack to get numbers instead - should probably raise an error
        class_names = range(100)
        # Get the predicted probabilities
    probs = model.predict_proba(data)
    # Use a masked array to check which predictions are correct, and then which we're most confident in
    class_exemplars = (
        np.ma.masked_array(
            probs,
            mask=np.repeat(model.predict(data) != y_data, probs.shape[1])
            # Need to repeat so the mask is the same shape as predict_proba
        )
        .argmax(0)
        .tolist()
    )
    # print(class_exemplars)
    for i, (class_index, class_name) in enumerate(zip(class_exemplars, class_names)):
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()
        # exemplar_data = df_test.iloc[class_index, :]
        exemplar_data = data[class_index, :]
        # Create the force plot
        fig = shap.force_plot(
            explainer.expected_value[i],
            shap_values[i][class_index],
            exemplar_data,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            text_rotation=30,
        )
        # Need to add label/text on the side for the class name
        print(f"{pretty_names(model_name, 'model')}")
        fig.suptitle(
            f"SHAP Force Plot for top exemplar using {pretty_names(model_name, 'model')} with class "
            + f"{class_name}",
            fontsize=16,
            y=1.4,
        )
        # Save the plot
        if save:
            fname = f"{experiment_folder / 'graphs' / 'shap_force_single'}_{model_name}_class{class_name}"
            fname += "_holdout" if holdout else ""
            save_fig(fig, fname)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()


def summary_SHAPdotplot_perclass(
    experiment_folder,
    class_names,
    model_name,
    feature_names,
    num_top,
    exemplar_X_test,
    exemplars_selected,
    data_forexplanations,
    data_indx,
    holdout=False,
):
    omicLogger.debug("Creating summary_SHAPdotplot_perclass...")

    if model_name == "xgboost" and len(class_names) == 2:
        print("Shape exemplars_selected: " + str(exemplars_selected.shape))
        class_name = class_names[1]
        print("Class: " + str(class_name))
        if holdout:
            fname = f"{experiment_folder / 'graphs' / 'summary_SHAPdotplot_perclass'}_{model_name}_{class_name}_holdout"
        else:
            fname = (
                f"{experiment_folder / 'graphs' / 'summary_SHAPdotplot_perclass'}_{data_forexplanations}_"
                + f"{model_name}_{class_name}"
            )

        # Plot shap bar plot
        shap.summary_plot(
            exemplars_selected,
            exemplar_X_test,
            plot_type="dot",
            color_bar="000",
            max_display=num_top,
            feature_names=feature_names,
            show=False,
        )
        fig = plt.gcf()
        save_fig(fig, fname)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()

        if not holdout:
            fname = f"{experiment_folder / 'results' / 'shapley_values'}_{data_forexplanations}_{model_name}"
            # saving the shapley values to dataframe
            df_shapley_values = pd.DataFrame(
                data=exemplars_selected, columns=feature_names, index=data_indx
            )
            # df_shapley_values.sort_index(inplace=True)
            df_shapley_values.index.name = "SampleID"
            df_shapley_values.to_csv(fname + ".csv")

    else:
        for i in range(len(class_names)):
            class_name = class_names[i]
            print("Class: " + str(class_name))
            print("i:" + str(i))

            print("Length exemplars_selected: " + str(len(exemplars_selected)))
            print("Type exemplars_selected: " + str(type(exemplars_selected)))

            if not holdout:
                fname_df = (
                    f"{experiment_folder / 'results' / 'shapley_values'}_{data_forexplanations}_{model_name}_"
                    + f"{class_name}_{i}"
                )
                # saving the shapley values to dataframe
                df_shapley_values = pd.DataFrame(
                    data=exemplars_selected[i], columns=feature_names, index=data_indx
                )
                # df_shapley_values.sort_index(inplace=True)
                df_shapley_values.index.name = "SampleID"
                df_shapley_values.to_csv(fname_df + ".csv")

            if holdout:
                fname = (
                    f"{experiment_folder / 'graphs' / 'summary_SHAPdotplot_perclass'}_{model_name}_{class_name}_"
                    + f"{'holdout'}_{i}"
                )
            else:
                fname = (
                    f"{experiment_folder / 'graphs' / 'summary_SHAPdotplot_perclass'}_{data_forexplanations}_"
                    + f"{model_name}_{class_name}_{i}"
                )

            # Plot shap bar plot
            shap.summary_plot(
                exemplars_selected[i],
                exemplar_X_test,
                plot_type="dot",
                color_bar="000",
                max_display=num_top,
                feature_names=feature_names,
                show=False,
            )
            my_cmap = plt.get_cmap("viridis")

            # Change the colormap of the artists
            for fc in plt.gcf().get_children():
                for fcc in fc.get_children():
                    if hasattr(fcc, "set_cmap"):
                        fcc.set_cmap(my_cmap)
            fig = plt.gcf()
            save_fig(fig, fname)
            plt.draw()
            plt.tight_layout()
            plt.pause(0.001)
            time.sleep(2)
            # Close the figure to ensure we start anew
            plt.clf()
            plt.close()

    plt.clf()
    plt.close()


def get_exemplars(x_test, y_test, model, problem_type, pcAgreementLevel):
    # Get the predictions
    pred_y = model.predict(x_test).flatten()

    test_y = y_test

    # Create empty array of indices
    exemplar_indices = []

    # Handle classification and regression differently

    # Classification
    if problem_type == CLASSIFICATION:
        print(CLASSIFICATION)
        # Return indices of equal elements between two arrays
        exemplar_indices = np.equal(pred_y, test_y)

    # Regression
    elif problem_type == REGRESSION:
        print("Regression - Percentage Agreement Level:", pcAgreementLevel)

        if pcAgreementLevel == 0:
            absPcDevArr = np.abs((np.divide(np.subtract(pred_y, test_y), test_y) * 100))
            exemplar_indices = absPcDevArr == pcAgreementLevel
        else:
            absPcDevArr = np.abs((np.divide(np.subtract(pred_y, test_y), test_y) * 100))
            exemplar_indices = absPcDevArr < pcAgreementLevel

    # create dataframe for exemplars
    exesToShow = []
    i = 0
    for val in exemplar_indices:
        if val is True:
            exesToShow.append({"idx": i, "testVal": test_y[i], "predVal": pred_y[i]})
        i = i + 1

    # create array with exemplars
    exemplar_X_test = []
    for row in exesToShow:
        exemplar_X_test.append(x_test[int(row["idx"])])
    exemplar_X_test = np.array(exemplar_X_test)

    return exemplar_X_test


def compute_average_abundance_top_features(
    problem_type,
    num_top,
    model_name,
    class_names,
    feature_names,
    data,
    shap_values_selected,
):
    # Get the names of the features
    names = feature_names

    # Create a dataframe to get the average abundance of each feature
    dfMaster = pd.DataFrame(data, columns=names)
    print(dfMaster.head())

    # Deal with classification differently, classification has shap values for each class
    # Get the SHAP values (global impact) sorted from the highest to the lower (absolute value)
    if problem_type == CLASSIFICATION:
        # XGBoost for binary classification seems to return the SHAP values only for class 1
        if model_name == "xgboost" and len(class_names) == 2:
            feature_order = np.argsort(np.mean(np.abs(shap_values_selected), axis=0))
            shap_values_mean_sorted = np.flip(
                np.sort(np.mean(np.abs(shap_values_selected), axis=0))
            )
        # When class > 2 (or class > 1 for all the models except XGBoost) SHAP return a list of SHAP value matrices.
        # One for each class.
        else:
            print(type(shap_values_selected))
            print(len(shap_values_selected))

            shap_values_selected_class = []
            for i in range(len(shap_values_selected)):
                print("Class: " + str(i))
                shap_values_selected_class.append(
                    np.mean(np.abs(shap_values_selected[i]), axis=0)
                )
            a = np.array(shap_values_selected_class)
            a_mean = np.mean(a, axis=0)
            feature_order = np.argsort(a_mean)
            shap_values_mean_sorted = np.flip(np.sort(a_mean))

    # Deal with regression
    else:
        # Get the SHAP values (global impact) sorted from the highest to the lower (absolute value)
        feature_order = np.argsort(np.mean(np.abs(shap_values_selected), axis=0))
        shap_values_mean_sorted = np.flip(
            np.sort(np.mean(np.abs(shap_values_selected), axis=0))
        )

    # In all cases flip feature order anyway to agree with shap_values_mean_sorted
    feature_order = np.flip(feature_order)

    # Select names, average abundance of top features
    top_names = []
    top_averageAbund = []

    if num_top < len(feature_order):
        lim = num_top
    else:
        lim = len(feature_order)

    for j in range(0, lim):
        i = feature_order[j]
        top_names.append(names[i])

        # Get the average of abundance across all the samples not only exemplar
        abund = np.mean(dfMaster[names[i]])
        top_averageAbund.append(abund)

    # Return everything - only SHAP values for the top features
    print("TOP NAMES: ")
    print(top_names)

    print("TOP ABUNDANCE: ")
    print(top_averageAbund)

    return top_names, top_averageAbund, shap_values_mean_sorted[:num_top]


def shap_plots(
    experiment_folder,
    problem_type,
    model_list,
    explanations_data,
    feature_names,
    x,
    x_test,
    y_test,
    x_train,
    num_top_features,
    pcAgreementLevel=10,
    save=True,
    holdout=False,
):
    omicLogger.debug("Creating shap_plots...")

    if explanations_data == "all" or "test" or "train" or "exemplars":
        data_forexplanations = explanations_data
    # assume test set
    else:
        data_forexplanations = "train"

    if len(feature_names) <= num_top_features:
        num_top = len(feature_names)
    else:
        num_top = num_top_features

    # Convert the data into dataframes to ensure features are displayed
    df_train = pd.DataFrame(data=x_train, columns=feature_names)
    print(feature_names)
    print(len(feature_names))

    # Loop over the defined models
    for model_name in model_list:
        # Load the model
        model_path = get_model_path(experiment_folder, model_name)

        print("Model path")
        print(model_path)
        print("Model name")
        print(model_name)

        print(f"Plotting SHAP plots for {model_name}")
        omicLogger.info(f"Plotting SHAP plots for {model_name}")

        model = utils.load.load_model(model_name, model_path)

        # Select the right explainer from SHAP
        explainer = select_explainer(model, model_name, df_train, problem_type)

        # Get the exemplars on the test set -- maybe to modify to include probability
        exemplar_X_test = get_exemplars(
            x_test, y_test, model, problem_type, pcAgreementLevel
        )

        shap_values, data, data_indx = compute_shap_vals(
            experiment_folder,
            data_forexplanations,
            explainer,
            x,
            x_train,
            x_test,
            exemplar_X_test,
        )
        # Handle regression and classification differently and store the shap_values in shap_values_selected

        # Classification
        if problem_type == CLASSIFICATION:
            objects, abundance, shap_values_mean_sorted = shap_plot_clf(
                shap_values,
                model,
                model_name,
                data,
                data_indx,
                num_top,
                feature_names,
                experiment_folder,
                data_forexplanations,
                holdout,
                save,
            )

        # Regression
        else:
            objects, abundance, shap_values_mean_sorted = shap_plot_reg(
                model_name,
                shap_values,
                holdout,
                save,
                experiment_folder,
                data_forexplanations,
                feature_names,
                data_indx,
                data,
                num_top,
            )

        # Displaying the average percentage %
        abundance = np.asarray(abundance) / 10

        d = {
            "Features": objects,
            "SHAP values": shap_values_mean_sorted,
            "Average abundance": list(abundance),
        }

        fname = (
            f"{experiment_folder / 'results' / 'top_features_AbsMeanSHAP_Abundance'}_{data_forexplanations}_"
            + f"{model_name}"
        )
        fname += "_holdout" if holdout else ""
        df = pd.DataFrame(d)
        df.to_csv(fname + ".csv")

        # Bar plot of average abundance across all the samples of the top genera

        y_pos = np.arange(len(objects))
        plt.barh(y_pos, abundance, align="center", color="black")
        plt.yticks(y_pos, objects)
        plt.gca().invert_yaxis()
        plt.xlabel("Average abundance (%)")

        fig = plt.gcf()

        if save:
            fname = (
                f"{experiment_folder / 'graphs' / 'abundance_top_features_exemplars'}_{data_forexplanations}_"
                + f"{model_name}"
            )
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


def shap_summary_plot(
    experiment_folder,
    model_list: list[str],
    problem_type: str,
    x_test,
    feature_names,
    shap_dict,
    save: bool = True,
    holdout: bool = False,
):
    """
    A wrapper to prepare the data and models for the SHAP summary plot
    """
    omicLogger.debug("Creating shap_summary_plot...")
    # Convert the data into dataframes to ensure features are displayed
    df_test = pd.DataFrame(data=x_test, columns=feature_names)
    # Get the model paths
    for model_name in model_list:
        model_path = get_model_path(experiment_folder, model_name)

        print(f"Plotting SHAP for {model_name}")
        model = utils.load.load_model(model_name, model_path)
        # Define the figure object
        fig, ax = plt.subplots()
        # Select the right explainer from SHAP
        shap_dict[model_name][0]
        # Calculate the shap values
        shap_values = shap_dict[model_name][1]
        # Handle regression and classification differently
        if problem_type == CLASSIFICATION:
            # Try to get the class names
            try:
                class_names = model.classes_.tolist()
            except AttributeError:
                print(
                    "Unable to get class names automatically - classes will be encoded"
                )
                class_names = None

            # Use SHAP's summary plot
            shap.summary_plot(
                shap_values,
                df_test,
                plot_type="violin",
                show=False,
                class_names=class_names,
            )
        elif problem_type == REGRESSION:
            shap.summary_plot(shap_values, df_test, plot_type="violin", show=False)
        # Get the figure object
        fig = plt.gcf()
        if save:
            fname = f"{experiment_folder / 'graphs' / 'shap_summary'}_{model_name}"
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


def compute_shap_vals(
    experiment_folder,
    data_forexplanations,
    explainer,
    x,
    x_train,
    x_test,
    exemplar_X_test,
):
    # Compute SHAP values for desired data (either test set x_test, or exemplar_X_test or the entire dataset x)
    data_indx = pd.read_csv(
        experiment_folder / "transformed_model_input_data.csv",
        index_col=0,
        usecols=["SampleID", "set"],
    )
    if data_forexplanations == "all":
        shap_values = explainer.shap_values(x)
        data = x
        data_indx = data_indx.index

    elif data_forexplanations == "train":
        shap_values = explainer.shap_values(x_train)
        data = x_train
        data_indx = data_indx[data_indx.set == "Train"].index

    elif data_forexplanations == "test":
        shap_values = explainer.shap_values(x_test)
        data = x_test
        data_indx = data_indx[data_indx.set == "Test"].index

    elif data_forexplanations == "exemplars":
        shap_values = explainer.shap_values(exemplar_X_test)
        data = exemplar_X_test
        data_indx = data_indx[data_indx.set == "Test"].index

    # otherwise assume train set
    else:
        shap_values = explainer.shap_values(x_train)
        data = x_train
        data_indx = data_indx[data_indx.set == "Train"].index

    return shap_values, data, data_indx


def shap_plot_clf(
    shap_values,
    model,
    model_name,
    data,
    data_indx,
    num_top,
    feature_names,
    experiment_folder,
    data_forexplanations,
    holdout,
    save,
):
    # For classification there is not difference between data structure returned by SHAP
    shap_values_selected = shap_values

    # Try to get the class names
    try:
        class_names = model.classes_.tolist()
    except AttributeError:
        print("Unable to get class names automatically - classes will be encoded")
        class_names = None

    # Produce and save SHAP bar plot

    if model_name == "xgboost" and len(class_names) == 2:
        # Use SHAP's summary plot
        shap.summary_plot(
            shap_values_selected,
            data,
            plot_type="bar",
            max_display=num_top,
            color=plt.get_cmap("Set3"),
            feature_names=feature_names,
            show=False,
            class_names=class_names,
        )
        fig = plt.gcf()

        # Save the plot for multi-class classification
        if save:
            fname = f"{experiment_folder / 'graphs' / 'shap_bar_plot'}_{data_forexplanations}_{model_name}"
            fname += "_holdout" if holdout else ""
            save_fig(fig, fname)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()

    else:
        # Use SHAP's summary plot
        shap.summary_plot(
            shap_values_selected,
            data,
            plot_type="bar",
            max_display=num_top,
            feature_names=feature_names,
            show=False,
            class_names=class_names,
        )
        fig = plt.gcf()

        # Save the plot for multi-class classification
        if save:
            fname = f"{experiment_folder / 'graphs' / 'shap_bar_plot'}_{data_forexplanations}_{model_name}"
            fname += "_holdout" if holdout else ""
            save_fig(fig, fname)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)
        time.sleep(2)
        # Close the figure to ensure we start anew
        plt.clf()
        plt.close()

    (
        objects,
        abundance,
        shap_values_mean_sorted,
    ) = compute_average_abundance_top_features(
        CLASSIFICATION,
        num_top,
        model_name,
        class_names,
        feature_names,
        data,
        shap_values_selected,
    )

    summary_SHAPdotplot_perclass(
        experiment_folder,
        class_names,
        model_name,
        feature_names,
        num_top,
        data,
        shap_values_selected,
        data_forexplanations,
        data_indx,
        holdout,
    )

    plt.clf()
    plt.close()
    # Clear keras and TF sessions/graphs etc.
    K.clear_session()

    return objects, abundance, shap_values_mean_sorted


def shap_plot_reg(
    model_name,
    shap_values,
    holdout,
    save,
    experiment_folder,
    data_forexplanations,
    feature_names,
    data_indx,
    data,
    num_top,
):
    # Produce and save bar plot for regression

    # Handle Shap saves differently the values for Keras when it's regression
    if model_name == "mlp_keras":
        shap_values_selected = shap_values[0]
    else:
        shap_values_selected = shap_values

    if not holdout:
        fname = f"{experiment_folder / 'results' / 'shapley_values'}_{data_forexplanations}_{model_name}"
        # saving the shapley values to dataframe
        df_shapley_values = pd.DataFrame(
            data=shap_values_selected, columns=feature_names, index=data_indx
        )
        # df_shapley_values.sort_index(inplace=True)
        df_shapley_values.index.name = "SampleID"
        df_shapley_values.to_csv(fname + ".csv")

    # Plot shap bar plot
    shap.summary_plot(
        shap_values_selected,
        data,
        plot_type="bar",
        color_bar="000",
        max_display=num_top,
        feature_names=feature_names,
        show=False,
    )
    fig = plt.gcf()

    # Save the plot
    if save:
        fname = f"{experiment_folder / 'graphs' / 'shap_bar_plot'}_{data_forexplanations}_{model_name}"
        fname += "_holdout" if holdout else ""
        save_fig(fig, fname)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
    time.sleep(2)
    # Close the figure to ensure we start anew
    plt.clf()
    plt.close()
    # Clear keras and TF sessions/graphs etc.
    K.clear_session()

    #  #Produce and save dot plot for regression

    shap.summary_plot(
        shap_values_selected,
        data,
        plot_type="dot",
        color_bar="000",
        max_display=num_top,
        feature_names=feature_names,
        show=False,
    )
    fig = plt.gcf()
    # Save the plot
    if save:
        fname = f"{experiment_folder / 'graphs' / 'shap_dot_plot'}_{data_forexplanations}_{model_name}"
        fname += "_holdout" if holdout else ""
        save_fig(fig, fname)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
    time.sleep(2)

    # Close the figure to ensure we start anew
    plt.clf()
    plt.close()

    # Clear keras and TF sessions/graphs etc.
    K.clear_session()

    # Plot abundance bar plot feature from SHAP
    class_names = []
    (
        objects,
        abundance,
        shap_values_mean_sorted,
    ) = compute_average_abundance_top_features(
        REGRESSION,
        num_top,
        model_name,
        class_names,
        feature_names,
        data,
        shap_values_selected,
    )

    return objects, abundance, shap_values_mean_sorted
