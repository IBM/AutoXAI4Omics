import pandas as pd
from omics import R_replacement as rrep
import joblib

# import logging # uncomment this when incorporated in AO
# omicLogger = logging.getLogger("OmicLogger") #uncommment this when incorporated in AO


def get_data_metabolomic(config_dict, holdout=False):
    """
    - Runs preprocessing_LO .
    - Filters metadata based on processed data (removes any samples removed during processing)
    - Returns x,y,feature_names

    Parameters
    ---------
    config_dict: config dictionary

    Returns
    --------
    x,y, feature names (in correct format format for ML)

    """

    # add filter_genes & filter_samples parameters from config_dict that have default values set in config
    filter_genes1 = int(config_dict["metabolomic"]["filter_measurements"][0])
    filter_genes2 = int(config_dict["metabolomic"]["filter_measurements"][1])
    filter_samples = config_dict["metabolomic"]["filter_metabolomic_sample"]

    # add the output file name from config_dict that is required
    if config_dict["metabolomic"]["output_file_met"] is not None:
        output_file = config_dict["metabolomic"]["output_file_met"]
        print("Output file: " + output_file)
    else:
        output_file = "processed_metabolomic_data"
    output_file += "_holdout" if holdout else ""

    # add metadata output file from config_dict that is required
    if config_dict["metabolomic"]["output_metadata"] is not None:
        metout_file = config_dict["metabolomic"]["output_metadata"]
    else:
        metout_file = "processed_metabolomic_metadata"
    metout_file += "_holdout" if holdout else ""

    # Based on GE data type, perform ge preprocessing (functions in preprocessing.py)
    # omicLogger.debug('Loading Gene Expression data...')

    filtered_data, genestokeep = rrep.preprocessing_LO(
        config_dict, filtergene1=filter_genes1, filtergene2=filter_genes2, filter_sample=filter_samples, holdout=holdout
    )
    print("data type = ", config_dict["data"]["data_type"])

    # Save filtered ge data
    filtered_data.to_csv(output_file)

    # save list of genes kept
    save_name = (
        f'/experiments/results/{config_dict["data"]["name"]}/omics_{config_dict["data"]["data_type"]}_keptGenes.pkl'
    )
    with open(save_name, "wb") as f:
        joblib.dump(genestokeep, f)

    # If metadata file is present (assume target in metadata), remove any samples removed during filtering, save as metout
    # and extract target from metadata. If metadata not present, assume target in data file.
    metafile = "metadata_file" + ("_holdout_data" if holdout else "")
    if (config_dict["data"][metafile] != "") and (config_dict["data"][metafile] is not None):
        metadata = pd.read_csv(config_dict["data"]["metadata_file"], index_col=0)
        mask = metadata.index.isin(filtered_data.index)
        filtered_metadata = metadata.loc[mask]
        filtered_metadata.to_csv(metout_file)
        y = filtered_metadata[config_dict["data"]["target"]].values

    else:
        file = "file_path" + ("_holdout_data" if holdout else "")
        unfiltered_data = pd.read_csv(config_dict["data"][file], index_col=0)
        target_y = unfiltered_data.loc[
            config_dict["data"]["target"]
        ]  # need loc because target in ge data is ROW not column (as in metadata)
        # Filter y
        mask = target_y.index.isin(filtered_data.index)
        filtered_target_y = target_y.loc[mask]
        y = filtered_target_y.values

    feature_names = filtered_data.columns

    return filtered_data, y, feature_names


def get_data_metabolomic_trained(config_dict, holdout=False, prediction=False):
    """
    - Runs preprocessing_LO function.
    - Filters metadata based on processed data (removes any samples removed during processing)
    - Returns x,y,feature_names

    Parameters
    ---------
    config_dict: config dictionary

    Returns
    --------
    x,y, feature names (in correct format format for ML)

    """

    filtered_data = rrep.apply_learned_processing(config_dict, holdout=holdout, prediction=prediction)
    print("data type = ", config_dict["data"]["data_type"])

    # If metadata file is present (assume target in metadata), remove any samples removed during filtering, save as metout
    # and extract target from metadata. If metadata not present, assume target in data file.
    if holdout:
        metafile = "metadata_file" + ("_holdout_data" if holdout else "")
        if (config_dict["data"][metafile] != "") and (config_dict["data"][metafile] is not None):
            metadata = pd.read_csv(config_dict["data"][metafile], index_col=0)
            mask = metadata.index.isin(filtered_data.index)
            filtered_metadata = metadata.loc[mask]
            filtered_metadata.to_csv(metout_file)
            y = filtered_metadata[config_dict["data"]["target"]].values

        else:
            file = "file_path" + ("_holdout_data" if holdout else "")
            unfiltered_data = pd.read_csv(config_dict["data"][file], index_col=0)
            target_y = unfiltered_data.loc[
                config_dict["data"]["target"]
            ]  # need loc because target in ge data is ROW not column (as in metadata)
            # Filter y
            mask = target_y.index.isin(filtered_data.index)
            filtered_target_y = target_y.loc[mask]
            y = filtered_target_y.values
    else:
        y = None

    feature_names = filtered_data.columns

    return filtered_data, y, feature_names
