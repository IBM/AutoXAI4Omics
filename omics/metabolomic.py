import logging
omicLogger = logging.getLogger("OmicLogger")
import subprocess

def get_data_metabolomic(path_file, metadata_path, config_dict):
    '''
    Load and process the data
    '''
    omicLogger.debug('Loading Metabolic data...')
    # Use calour to create an experiment
    print("Path file: " +path_file)
    print("Metadata file: " +metadata_path)

    print("")
    print("")
    print("")
    print("***** Preprocessing metabolomic data *******")

    # add the input file parameter
    strcommand = "--expressionfile "+ path_file + " "

    # add the expression type parameter that is required
    expression_type = "MET"
    strcommand = strcommand+"--expressiontype "+expression_type+ " "

    # add the filter_samples parameter that is optional
    if config_dict["filter_metabolomic_sample"] is not None:
        filter_met_samples = config_dict["filter_metabolomic_sample"]
        print(filter_met_samples)
        strcommand = strcommand + "--Filtersamples " + str(filter_met_samples) + " "

    # add the filter_genes parameter to filter measurements that is optional
    if config_dict["filter_measurements"] is not None:
        filter_measurements = config_dict["filter_measurements"][0] +" "+config_dict["filter_measurements"][1]
        print(filter_measurements)
        strcommand = strcommand+"--Filtergenes "+filter_measurements+" "

    # add the output file name that is required
    if config_dict["output_file_met"] is not None:
        output_file_met = config_dict["output_file_met"]
        print(output_file_met)
        strcommand = strcommand+"--output "+output_file_met+" "

    # add the metadata for filtering 
    strcommand = strcommand+"--metadatafile "+ metadata_path + " "

    #add metadata output file that is required
    if config_dict["output_metadata"] is not None:
        metout_file = config_dict["output_metadata"]
        print(metout_file)
        strcommand = strcommand+"--outputmetadata "+metout_file

    print(strcommand)
    omicLogger.debug('Running python/R preprocessing script...')
    python_command = "python omics/AoT_gene_expression_pre_processing.py "+strcommand
    print(python_command)
    subprocess.call(python_command, shell=True)
    from data_processing import get_data
    x,y, feature_names = get_data(config_dict["output_file_met"], config_dict["target"], metout_file)

    return x,y, feature_names

