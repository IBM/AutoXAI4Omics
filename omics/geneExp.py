import pandas as pd
from omics import R_replacement as rrep

#import logging # uncomment this when incorporated in AO
#omicLogger = logging.getLogger("OmicLogger") #uncommment this when incorporated in AO

def get_data_gene_expression(config_dict,holdout=False):

    """
    - Runs one of 3 gene expression preprocessing functions based on data type.
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
    filter_genes1 = int(config_dict['gene_expression']["filter_genes"][0])
    filter_genes2 = int(config_dict['gene_expression']["filter_genes"][1])
    filter_samples = config_dict['gene_expression']["filter_sample"]

    # add the output file name from config_dict that is required
    if config_dict['gene_expression']["output_file_ge"] is not None:
        output_file = config_dict['gene_expression']["output_file_ge"]
        print("Output file: " +output_file)
    else:
        output_file = "processed_gene_expression_data"
    output_file += "_holdout" if holdout else ""
    
    #add metadata output file from config_dict that is required 
    if config_dict['gene_expression']["output_metadata"] is not None:
        metout_file = config_dict['gene_expression']["output_metadata"]
    else:
        metout_file = "processed_gene_expression_metadata"
    metout_file+= "_holdout" if holdout else ""
    
    #Based on GE data type, perform ge preprocessing (functions in preprocessing.py)
    #omicLogger.debug('Loading Gene Expression data...')

    if config_dict['gene_expression']["expression_type"] == "COUNTS":
        filtered_data = rrep.preprocessing_TMM(config_dict,filtergene1=filter_genes1, filtergene2=filter_genes2, filter_sample=filter_samples,holdout=holdout)
        print("data type = ", config_dict['gene_expression']["expression_type"])

    elif config_dict['gene_expression']["expression_type"] == "FPKM" or config_dict['gene_expression']["expression_type"] == "RPKM" or config_dict['gene_expression']["expression_type"] == "TPM" or config_dict['gene_expression']["expression_type"] == "TMM":
        filtered_data = rrep.preprocessing_others(config_dict,filtergene1=filter_genes1, filtergene2=filter_genes2, filter_sample=filter_samples,holdout=holdout)
        print("data type = ", config_dict['gene_expression']["expression_type"])

    elif config_dict['gene_expression']["expression_type"] == "Log2FC" or config_dict['gene_expression']["expression_type"] == "OTHER" or config_dict['gene_expression']["expression_type"] == "MET" or config_dict['gene_expression']["expression_type"] == "TAB":
        filtered_data = rrep.preprocessing_LO(config_dict,filtergene1=filter_genes1, filtergene2=filter_genes2, filter_sample=filter_samples,holdout=holdout)
        print("data type = ", config_dict['gene_expression']["expression_type"])

    else: # it's defined as 'OTHER'
        filtered_data = rrep.preprocessing_LO(config_dict,filtergene1=filter_genes1, filtergene2=filter_genes2, filter_sample=filter_samples,holdout=holdout)
        print("data type = ", config_dict['gene_expression']["expression_type"])


    #Save filtered ge data
    filtered_data.to_csv(output_file) 
    
    #If metadata file is present (assume target in metadata), remove any samples removed during filtering, save as metout
    #and extract target from metadata. If metadata not present, assume target in data file.
    metafile = "metadata_file"+ ("_holdout_data" if holdout else "")
    if (config_dict['data'][metafile] != ""):
        metadata = pd.read_csv(config_dict['data']["metadata_file"], index_col=0)  
        mask = metadata.index.isin(filtered_data.index)  
        filtered_metadata = metadata.loc[mask]
        filtered_metadata.to_csv(metout_file) 
        y = filtered_metadata[config_dict['data']["target"]].values
     
    else:
        file = "file_path"+ ("_holdout_data" if holdout else "")
        unfiltered_data = pd.read_csv(config_dict['data'][file], index_col=0)
        target_y = unfiltered_data.loc[config_dict['data']["target"]] # need loc because target in ge data is ROW not column (as in metadata)
        # Filter y 
        mask = target_y.index.isin(filtered_data.index) 
        filtered_target_y = target_y.loc[mask]
        y = filtered_target_y.values

    feature_names = filtered_data.columns

    return filtered_data,y, feature_names    