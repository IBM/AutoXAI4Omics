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

from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from typing import Union
import calour as ca
import joblib
import logging
import numpy as np
import pandas as pd
import scipy.sparse

omicLogger = logging.getLogger("OmicLogger")


def create_microbiome_calourexp(
    fpath_biom: Union[Path, str],
    fpath_meta: Union[Path, str],
    norm_reads: Union[float, int, None] = 1000,
    min_reads: Union[float, int, None] = 1000,
) -> ca.AmpliconExperiment:
    """
    Create the experiment from calour using the given minimum number of reads and the number of reads to normalize to
    """
    try:
        exp = ca.read_amplicon(
            data_file=fpath_biom,
            sample_metadata_file=fpath_meta,
            normalize=norm_reads,
            min_reads=min_reads,
        )
    except Exception as e:
        omicLogger.error("error with reading amplicon")
        raise e

    return exp


def filter_biom(
    config_dict: dict,
    amp_exp: ca.AmpliconExperiment,
    abundance: Union[float, int] = 10,
    prevalence: Union[float, int] = 0.01,
    collapse_tax=None,
) -> ca.AmpliconExperiment:
    """
    Filter the biom data using the given abudance and prevalance

    Can also collapse the taxonomy if given (uses default species otherwise)
    """

    omicLogger.info(f"Original data size: {amp_exp.data.shape}")
    # Filter abundance
    amp_exp = amp_exp.filter_sum_abundance(abundance)
    # Filter prevalence
    amp_exp = amp_exp.filter_prevalence(prevalence)

    # save list of ml features kept
    featureToKeep = list(amp_exp.feature_metadata["_feature_id"])
    save_name = f'/experiments/results/{config_dict["data"]["name"]}/omics_{config_dict["data"]["data_type"]}_keptFeatures.pkl'
    with open(save_name, "wb") as f:
        joblib.dump(featureToKeep, f)

    # Collapse taxonomy
    if collapse_tax is not None:
        try:
            nrows, _ = amp_exp.data.shape
            amp_exp = amp_exp.collapse_taxonomy(collapse_tax)
            nrows_after, _ = amp_exp.data.shape
            if nrows_after < nrows:
                omicLogger.info(
                    f"Reduced number of samples from {nrows} to {nrows_after}"
                )
        except ValueError:
            omicLogger.info(
                f"{collapse_tax} is not a valid taxonomy level, must be 'k', 'p', 'c', 'o', 'f', 'g', or 's'"
            )
            raise
    omicLogger.info(
        "Data dimension after collapsing and filtering OTUs by abundance and prevalence: "
        + str(amp_exp.data.shape)
    )

    return amp_exp


def filter_samples(
    amp_exp: ca.AmpliconExperiment, filter_obj: dict
) -> ca.AmpliconExperiment:
    """
    Filter the metadata samples using a {col_name: [remove_vals]} construct

    If the provided structure is a list, then we loop over this list and make successive calls
    """
    if isinstance(filter_obj, dict):
        for col_name, filter_vals in filter_obj.items():
            amp_exp = filter_metadata(amp_exp, col_name, filter_vals)
    elif isinstance(filter_obj, list):
        amp_exp = filter_multiple(amp_exp, filter_obj)
    return amp_exp


def filter_multiple(
    amp_exp: ca.AmpliconExperiment,
    filter_list: list[dict],
    axis: int = 0,
    negate: bool = True,
) -> ca.AmpliconExperiment:
    # Loop over the filter dicts so that we can filter by multiple sets of conditions
    for filter_dict in filter_list:
        # We take our dataframe, and select only the columns we are looking at
        # Then we check which rows match the conditions
        # Then we use all (by row, or axis, 1) to select the rows where all conditions are met (all True)
        # Use .values to get the boolean array
        select = (
            amp_exp.sample_metadata[filter_dict.keys()].isin(filter_dict).all(1).values
        )
        # If negate is true, we want to discard/filter the rows which match our conditions
        if negate is True:
            select = ~select
        # Reorder the data and metadata
        amp_exp = amp_exp.reorder(select, axis=axis, inplace=False)
    return amp_exp


def filter_metadata(
    amp_exp: ca.AmpliconExperiment, col_name: str, to_filter
) -> ca.AmpliconExperiment:
    return amp_exp.filter_by_metadata(
        field=col_name, select=to_filter, axis=0, negate=True
    )


def modify_classes(
    amp_exp: ca.AmpliconExperiment, class_col_name, remove_class=None, merge_by=None
):
    """
    Helper function to merge and/or remove classes
    """
    if remove_class is None and merge_by is None:
        omicLogger.info(
            "No filtering or merging has been specified - nothing has changed!"
        )
        return amp_exp
    if remove_class is not None:
        amp_exp = filter_metadata(amp_exp, class_col_name, to_filter=remove_class)
    if merge_by is not None:
        amp_exp = merge_classes(amp_exp, class_col_name, merge_by=merge_by)
    return amp_exp


def merge_classes(amp_exp: ca.AmpliconExperiment, class_col_name, merge_by):
    # Get the relevant class column
    class_col = amp_exp.sample_metadata[class_col_name]
    # Loop through the merge_class dict and replace
    for new_class, old_classes in merge_by.items():
        class_col = class_col.replace(to_replace=old_classes, value=new_class)
    # Insert new class column
    amp_exp.sample_metadata[class_col_name] = class_col
    return amp_exp


def prepare_data(amp_exp: ca.AmpliconExperiment):
    """
    Extract data from calour experiment and transform using StandardScaler
    """
    if scipy.sparse.issparse(amp_exp.data):
        data = amp_exp.data.todense()
    else:
        data = amp_exp.data

    data = np.asarray(data)

    SS = StandardScaler()
    data = SS.fit_transform(data)

    return data, SS


def select_class_col(
    amp_exp: ca.AmpliconExperiment, encoding=None, index=None, name=None
):
    """
    Selects the class column from the metadata either by the index or by name
    """
    # Need to provide at least one argument
    if index is None and name is None:
        raise ValueError("At least one argument must be provided")
    # Cannot select by both index and name
    elif index is not None and name is not None:
        raise ValueError("Only 'index' or 'name' can be selected")

    # Select the column either by index or name
    if index is not None:
        y = amp_exp.sample_metadata.iloc[:, index]
    elif name is not None:
        y = amp_exp.sample_metadata[name]

    # One-hot encoding
    if encoding == "onehot":
        enc = OneHotEncoder(sparse_output=False)
        y = enc.fit_transform(y.values.reshape(-1, 1))
        omicLogger.info(f"Categories using one-hot encoding {enc.categories_}")

    # Label encoding
    elif encoding == "label":
        enc = LabelEncoder()
        y = enc.fit_transform(y.values)
        omicLogger.info(f"Categories using label encoding {enc.classes_}")
        code = enc.transform(enc.classes_)
        omicLogger.info(f"Corresponding encoding {code}")

    return y


def get_feature_names_calourexp(
    amp_exp: ca.AmpliconExperiment, config_dict: dict
) -> list[str]:
    """
    Get (unique) feature names from the feature metadata to use in e.g. SHAP
    """
    if config_dict["collapse_tax"] == "species":
        feature_names = amp_exp.feature_metadata["taxonomy"].apply(
            lambda x: " ".join([x.split(";")[-2], x.split(";")[-1]])
            .replace("[", "")
            .replace("]", "")
        )
    else:
        feature_names = amp_exp.feature_metadata["_feature_id"].apply(
            lambda x: x.split(";")[-1].replace("[", "").replace("]", "")
        )
    # If we have duplicates
    if feature_names.value_counts().max() > 1:
        # Get the value counts
        counts = feature_names.value_counts()
        # Get those that are duplicated only
        dup_features = counts[counts > 1]
        # Create a counter dict for them
        counter = {k: 1 for k in dup_features.index.tolist()}
        # Loop over our feature names
        for i, name in feature_names.items():
            # If this is one of the duplicates
            if name in counter:
                # To access the dict
                root_name = name
                # Create the new name
                name += f"_{counter[name]}"
                # Replace the old name
                feature_names.loc[i] = name
                # Increment the counter to get a unique name
                counter[root_name] += 1
    feature_names.to_list()
    return feature_names


def get_feature_names_alternative(amp_exp: ca.AmpliconExperiment) -> list[str]:
    inputFeatureData = amp_exp.feature_metadata
    """ Get simple names for taxonomy """
    taxons = inputFeatureData["taxonomy"]
    names = []
    for taxon in taxons:
        tmpSplit = str.split(taxon, ";")
        name = tmpSplit[len(tmpSplit) - 1]
        if "unresolved" in name:
            name = tmpSplit[len(tmpSplit) - 3] + tmpSplit[len(tmpSplit) - 1]
        names.append(name)
    return names


def get_feature_names_for_abundance(amp_exp: ca.AmpliconExperiment) -> list[str]:
    inputFeatureData = amp_exp.feature_metadata
    """ Get simple names for taxonomy """
    taxons = inputFeatureData["taxonomy"]
    names = []
    for taxon in taxons:
        tmpSplit = str.split(taxon, ";")
        name = tmpSplit[len(tmpSplit) - 1]
        names.append(name)
    return names


def get_data_microbiome(
    path_file: Union[str, Path], metadata_path: Union[str, Path], config_dict: dict
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Load and process the data
    """
    microbiome_config = config_dict["microbiome"]

    omicLogger.debug("Loading Microbiome data...")
    # Use calour to create an experiment
    omicLogger.info("Path file: " + path_file)
    omicLogger.info("Metadata file: " + metadata_path)
    if (microbiome_config["norm_reads"] is None) and (
        microbiome_config["min_reads"] is None
    ):
        amp_exp = create_microbiome_calourexp(path_file, metadata_path, None, None)
    else:
        amp_exp = create_microbiome_calourexp(
            path_file,
            metadata_path,
            microbiome_config["norm_reads"],
            microbiome_config["min_reads"],
        )
    omicLogger.info("")
    omicLogger.info("")
    omicLogger.info("")
    omicLogger.info("***** Preprocessing microbiome data *******")

    omicLogger.info(f"Original data dimension: {amp_exp.data.shape}")
    # Use calour to filter the data

    amp_exp = filter_biom(
        config_dict,
        amp_exp,
        abundance=microbiome_config["filter_abundance"],
        prevalence=microbiome_config["filter_prevalence"],
        collapse_tax=microbiome_config["collapse_tax"],
    )
    omicLogger.info(
        f"After filtering contaminant, collapsing at genus and filtering by abundance: {amp_exp.data.shape}"
    )

    # Filter any data that needs it
    if microbiome_config["filter_microbiome_samples"] is not None:
        amp_exp = filter_samples(
            amp_exp, microbiome_config["filter_microbiome_samples"]
        )

    # Modify the classes if need be
    amp_exp = modify_classes(
        amp_exp,
        config_dict["data"]["target"],
        remove_class=microbiome_config["remove_classes"],
        merge_by=microbiome_config["merge_classes"],
    )

    omicLogger.info(f"After filtering samples: {amp_exp.data.shape}")

    omicLogger.info("Save experiment after filtering with name exp_filtered")
    amp_exp.save("biom_data_filtered" + config_dict["data"]["name"])
    omicLogger.info("****************************************************")
    omicLogger.info("")
    omicLogger.info("")
    omicLogger.info("")

    # Prepare data (load and normalize)
    x, SS = prepare_data(amp_exp)
    omicLogger.info(x.shape)

    # save normaliser
    save_name = f'/experiments/results/{config_dict["data"]["name"]}/omics_{config_dict["data"]["data_type"]}_scaler.pkl'
    with open(save_name, "wb") as f:
        joblib.dump(SS, f)

    # Select the labels
    ml_config: dict = config_dict["ml"]
    y = select_class_col(
        amp_exp,
        encoding=ml_config.get("encoding"),
        name=config_dict["data"]["target"],
    )

    features_names = get_feature_names_calourexp(amp_exp, microbiome_config)

    # Check the data and labels are the right size
    assert len(x) == len(y)

    x2 = pd.DataFrame(x, amp_exp.sample_metadata["_sample_id"], features_names)
    y2 = y.values
    return x2, y2, features_names


def apply_biom_filtering(config_dict, amp_exp, collapse_tax=None):
    """
    Filter the biom data using the given abudance and prevalance

    Can also collapse the taxonomy if given (uses default species otherwise)
    """

    omicLogger.info(f"Original data size: {amp_exp.data.shape}")

    # save list of genes kept
    save_name = f'/experiments/results/{config_dict["data"]["name"]}/omics_{config_dict["data"]["data_type"]}_keptFeatures.pkl'
    with open(save_name, "rb") as f:
        featureToKeep = joblib.load(f)

    #### check to see what features this set is missing
    missingFeatures = set(featureToKeep) - set(amp_exp.feature_metadata["_feature_id"])
    omicLogger.info(f"Missing Features: {missingFeatures}")

    #### keep the features that we want
    amp_exp.filter_by_metadata(
        field="_feature_id", select=featureToKeep, axis="f", inplace=True
    )

    # Collapse taxonomy
    if collapse_tax is not None:
        try:
            nrows, _ = amp_exp.data.shape
            amp_exp = amp_exp.collapse_taxonomy(collapse_tax)
            nrows_after, _ = amp_exp.data.shape
            if nrows_after < nrows:
                omicLogger.info(
                    f"Reduced number of samples from {nrows} to {nrows_after}"
                )
        except ValueError:
            omicLogger.info(
                f"{collapse_tax} is not a valid taxonomy level, must be 'k', 'p', 'c', 'o', 'f', 'g', or 's'"
            )
            raise
    omicLogger.info(
        "Data dimension after collapsing and filtering OTUs by abundance and prevalence: "
        + str(amp_exp.data.shape)
    )

    return amp_exp


def get_data_microbiome_trained(
    config_dict: dict, holdout: bool = False, prediction: bool = False
):
    """
    Load and process the data
    """

    if not holdout and not prediction:
        raise ValueError("One of holdout or prediction must be true")

    if holdout:
        path_file = config_dict["data"]["file_path_holdout_data"]
        metadata_path = config_dict["data"]["metadata_file_holdout_data"]
    elif prediction:
        path_file = config_dict["prediction"]["file_path"]
        metadata_path = config_dict["prediction"]["metadata_file"]

    microbiome_config = config_dict["microbiome"]

    omicLogger.debug("Loading Microbiome data...")
    # Use calour to create an experiment
    omicLogger.info("Path file: " + path_file)
    omicLogger.info("Metadata file: " + metadata_path)
    if (microbiome_config["norm_reads"] is None) and (
        microbiome_config["min_reads"] is None
    ):
        amp_exp = create_microbiome_calourexp(path_file, metadata_path, None, None)
    else:
        amp_exp = create_microbiome_calourexp(
            path_file,
            metadata_path,
            microbiome_config["norm_reads"],
            microbiome_config["min_reads"],
        )

    omicLogger.info("***** Preprocessing microbiome data *******")

    omicLogger.info(f"Original data dimension: {amp_exp.data.shape}")
    # Use calour to filter the data

    amp_exp = apply_biom_filtering(
        config_dict, amp_exp, collapse_tax=microbiome_config["collapse_tax"]
    )

    omicLogger.info(
        f"After filtering contaminant, collapsing at genus and filtering by abundance: {amp_exp.data.shape}"
    )

    # Modify the classes if need be
    amp_exp = modify_classes(
        amp_exp,
        config_dict["data"]["target"],
        remove_class=microbiome_config["remove_classes"],
        merge_by=microbiome_config["merge_classes"],
    )

    # load scaler
    save_name = f'/experiments/results/{config_dict["data"]["name"]}/omics_{config_dict["data"]["data_type"]}_scaler.pkl'
    with open(save_name, "rb") as f:
        SS = joblib.load(f)

    # get data array. NOTE data is in our 'normal' ml view, rows=samples, columns=features
    if scipy.sparse.issparse(amp_exp.data):
        data = amp_exp.data.todense()
    else:
        data = amp_exp.data

    data = np.asarray(data)

    # apply scaler
    x = SS.transform(data)

    if prediction:
        y = None
    else:
        ml_config: dict = config_dict["ml"]
        y = select_class_col(
            amp_exp,
            encoding=ml_config.get("encoding"),
            name=config_dict["data"]["target"],
        )

    features_names = get_feature_names_calourexp(amp_exp, microbiome_config)

    x2 = pd.DataFrame(x, amp_exp.sample_metadata["_sample_id"], features_names)

    return x2, y, features_names
