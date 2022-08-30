import logging
omicLogger = logging.getLogger("OmicLogger")
import subprocess

def get_data_gene_expression(path_file, metadata_path, config_dict):
    '''
    Load and process the data
    '''
    omicLogger.debug('Loading Gene Expression data...')
    # Use calour to create an experiment
    print("Path file: " +path_file)
    print("Metadata file: " +metadata_path)

    print("")
    print("")
    print("")
    print("***** Preprocessing gene expression data *******")

    # add the input file parameter
    strcommand = "--expressionfile "+ path_file + " "

    # add the expression type parameter that is required
    if config_dict["expression_type"] is not None:
        expression_type = config_dict["expression_type"]
        print(expression_type)
    else:
        expression_type = "OTHER"
    strcommand = strcommand+"--expressiontype "+expression_type+ " "

    # add the filter_samples parameter that is optional
    if config_dict["filter_sample"] is not None:
        filter_samples = config_dict["filter_sample"]
        print(filter_samples)
        strcommand = strcommand + "--Filtersamples " + str(filter_samples) + " "

    # add the filter_genes parameter that is optional
    if config_dict["filter_genes"] is not None:
        filter_genes = config_dict["filter_genes"][0] +" "+config_dict["filter_genes"][1]
        print(filter_genes)
        strcommand = strcommand+"--Filtergenes "+filter_genes+" "

    # add the output file name that is required
    if config_dict["output_file_ge"] is not None:
        output_file = config_dict["output_file_ge"]
        print(output_file)
    else:
        output_file = "processed_gene_expression_data"
    strcommand = strcommand+"--output "+output_file+" "
        
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
    x,y, feature_names = get_data(config_dict["output_file_ge"], config_dict["target"], metout_file)

    return x,y, feature_names

