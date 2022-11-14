
import pandas as pd
from bioinfokit.analys import norm
import conorm as cn


def preprocessing_LO(config_dict,filtergene1,filtergene2,filter_sample,holdout):

    """
    Note - this function is replacing Run_LO.R
    ---------

    Parameters
    -------
    config_dict: config dictionary 
    filtergene1 & 2 : filtering paramters : keep genes > 'filtergene1' expression in "filtergene2" or more samples
                    : Default values are set filtergene1=0 (default expression) and filtergene2=1 (default # samples). 
    filter_sample : filtering parameter : remove samples with the number of expressed genes above or below 'filter_sample' SD's from the mean

    Returns
    --------
 
    Reads the input gene expression data and parameters and returns:
        i) data_final : gene expression data with filtered genes and samples removed

    """
    #omicLogger.debug('Filtering gene expression data...')
    file = "file_path"+ ("_holdout_data" if holdout else "")
    data_file = pd.read_csv(config_dict['data'][file], index_col=0) #sampleID as index

   #If metadata not provided, drop target prior to filtering
    metafile = "metadata_file"+ ("_holdout_data" if holdout else "")
    if(config_dict['data'][metafile] == ""):
        data_drop = data_file.drop(config_dict['data']["target"], axis=0)
        data = data_drop.loc[(data_drop != 0).any(axis=1), (data_drop != 0).any(axis=0)] #Remove any samples/rows with all zeros 
    else:
        data = data_file.loc[(data_file != 0).any(axis=1), (data_file != 0).any(axis=0)] #Remove any samples/rows with all zeros 

    data_abs = data.abs()  #Remove -ve values (for filtering purposes only, will add back later)    

    #Filter genes (only keep genes with exp > filtergene1 in filtergene2 or more samples)
    filterG = ((data_abs > filtergene1).sum(axis=1)) # Per gene, # samples in which exp > filtergene1
    genestokeep = filterG.loc[filterG >= filtergene2].index.tolist() # GeneIDs of exp genes > filtergene2
    data_filtered = data.loc[genestokeep] 
  
    #Transpose data (samples rows, genes as columns)
    tdata_filtered = data_filtered.transpose()
    
    #Filter samples
    data_filtered_abs = tdata_filtered.apply(abs) # remove -ve values
    sample_means = (data_filtered_abs > 0).mean(axis=1)  # mean proportion of expressed genes per sample (axis=1 selects over the row (per sample))
    mean_all = sample_means.mean() # mean of "sample means"
    std_all = sample_means.std() # stdev of "sample means" 
    
    print("Average proportion of expressed genes (> 0) across all samples (Sample Mean avg.):" , mean_all)
    print("Standard Deviation of expressed genes (> 0) across all samples (Sample Stdev avg.):" , std_all)

    #Calculate user filter: number of SD's away from the mean (filter_sample)
    user_filter = (filter_sample * std_all)
    print("Stdev avg. * user filter :", user_filter)
    print("Range (Sample mean avg. +/- Stdev avg.) for filtering samples to keep: " , (mean_all - user_filter), "to" , (mean_all + user_filter))
    print("Sample means :", sample_means)

    #Caculate if the mean number of expressed genes in any sample is outside range
    #1. Find mean # genes per sample with expression >0, 
    #2. Keep samples with number of expressed genes i: HIGHER than the global mean minus user_filter (# SD's from mean) AND ii) lower than global mean plus user filter
    tokeep = (sample_means >= (mean_all - user_filter)) & (sample_means <= (mean_all + user_filter)) 
    
    #Filter original df, keeping those samples that satisfy reqs
    data_final = tdata_filtered.loc[tokeep]   # filter original df (already filtered for genes) (with +ve and -ve values), keeping those samples    

    #Return final data, discarded genes, and discarded samples
    return data_final

    

#---------------------------------------------------------------------------------------------------#


def preprocessing_others(config_dict,filtergene1,filtergene2,filter_sample,holdout):

    """ 
    Note - this function is replacing Run_others.R (note this is the same as Run_LO.R, but without converting to absolute values)
    ---------

    Parameters
    -------
    pathfile : gene expression csv (REQ)
    filtergene1 & 2 : filtering paramters : keep genes > 'filtergene1' expression in "filtergene2" or more samples
                    : Default values are set filtergene1=0 (default expression) and filtergene2=1 (default # samples). 
    filter_sample : filtering parameter : remove samples with the number of expressed genes above or below 'filter_sample' SD's from the mean

    Returns
    --------
 
    Reads the input gene expression data and parameters and returns:
        i) data_final : gene expression data with filtered genes and samples removed

    """
   
    #omicLogger.debug('Filtering gene expression data...')
    file = "file_path"+ ("_holdout_data" if holdout else "")
    data_file = pd.read_csv(config_dict['data'][file], index_col=0) #sampleID as index
   
   #If metadata not provided, drop target prior to filtering
    metafile = "metadata_file"+ ("_holdout_data" if holdout else "")
    if(config_dict['data'][metafile] == ""):
        data_drop = data_file.drop(config_dict['data']["target"], axis=0)
        data = data_drop.loc[(data_drop != 0).any(axis=1), (data_drop != 0).any(axis=0)] #Remove any samples/rows with all zeros 
    else:
        data = data_file.loc[(data_file != 0).any(axis=1), (data_file != 0).any(axis=0)] #Remove any samples/rows with all zeros 

    #Filter genes (only keep genes with exp > filtergene1 in filtergene2 or more samples)
    filterG = ((data > filtergene1).sum(axis=1)) # Per gene, # samples in which exp > filtergene1
    genestokeep = filterG.loc[filterG >= filtergene2].index.tolist() # GeneIDs of exp genes > filtergene2
    data_filtered = data.loc[genestokeep] 

    #Transpose data (samples rows, genes as columns)
    tdata_filtered = data_filtered.transpose()

    #Filter samples
    sample_means = (tdata_filtered > 0).mean(axis=1)  # mean proportion of expressed genes per sample (axis=1 selects over the row (per sample))
    mean_all = sample_means.mean() # mean of "sample means"
    std_all = sample_means.std() # stdev of "sample means" 
    
    print("Average proportion of expressed genes (> 0) across all samples (Sample Mean avg.):" , mean_all)
    print("Standard Deviation of expressed genes (> 0) across all samples (Sample Stdev avg.):" , std_all)

    #Calculate user filter: number of SD's away from the mean (filter_sample)
    user_filter = (filter_sample * std_all)
    print("Stdev avg. * user filter :", user_filter)
    print("Range (Sample mean avg. +/- Stdev avg.) for filtering samples to keep: " , (mean_all - user_filter), "to" , (mean_all + user_filter))
    print("Sample means :", sample_means)

    #Caculate if the mean number of expressed genes in any sample is outside range
    #1. Find mean # genes per sample with expression >0, 
    #2. Keep samples with number of expressed genes i: HIGHER than the global mean minus user_filter (# SD's from mean) AND ii) lower than global mean plus user filter
    tokeep = (sample_means >= (mean_all - user_filter)) & (sample_means <= (mean_all + user_filter)) 
    
    #Filter original df, keeping those samples that satisfy reqs
    data_final = tdata_filtered.loc[tokeep]   # filter original df (already filtered for genes) (with +ve and -ve values), keeping those samples    

    #Return final data, discarded genes, and discarded samples
    return data_final

#---------------------------------------------------------------------------------------------------#

def preprocessing_TMM(config_dict,filtergene1,filtergene2,filter_sample,holdout):

    """ 
    Note - this function is replacing Run_TMM.R
         - this function transforms to CPM values to filter genes
         - this function transforms data to TPM then CPM before sample filtering 

    # TO DO - check with Laura : can data be converted to TMM, then CPM, and then both gene and sample filtering done?
    ---------

    Parameters
    -------
    pathfile : gene expression csv (REQ)
    filtergene1 & 2 : filtering paramters : keep genes > 'filtergene1' expression in "filtergene2" or more samples
                    : Default values are set filtergene1=0 (default expression) and filtergene2=1 (default # samples). 
    filter_sample : filtering parameter : remove samples with the number of expressed genes above or below 'filter_sample' SD's from the mean

    Returns
    --------
 
    Reads the input gene expression data and parameters and returns:
        i) data_final : gene expression data with filtered genes and samples removed
 
    """

    #omicLogger.debug('Filtering gene expression data...')
    file = "file_path"+ ("_holdout_data" if holdout else "")
    data_file = pd.read_csv(config_dict['data'][file], index_col=0) #sampleID as index
   
   #If metadata not provided, drop target prior to filtering
    metafile = "metadata_file"+ ("_holdout_data" if holdout else "")
    if(config_dict['data'][metafile] == ""):
        data_drop = data_file.drop(config_dict['data']["target"], axis=0)
        data = data_drop.loc[(data_drop != 0).any(axis=1), (data_drop != 0).any(axis=0)] #Remove any samples/rows with all zeros 
    else:
        data = data_file.loc[(data_file != 0).any(axis=1), (data_file != 0).any(axis=0)] #Remove any samples/rows with all zeros 

    #Normalise data to CPM (for gene filtering)
    data_ins = norm() # create instance of norm class
    data_ins.cpm(df=data) # Run method cpm on instance (this creates class attribute 'cpm_norm)
    data_norm = data_ins.cpm_norm # assign instance attribute cpm_norm to data_norm

    #Filter genes using CPM data (only keep genes with CPM > filtergene1 in filtergene2 or more samples)
    filterG = ((data_norm > filtergene1).sum(axis=1)) # Per gene, # samples in which CPM > filtergene1
    genestokeep = filterG.loc[filterG >= filtergene2].index.tolist() # GeneIDs of CPM genes > filtergene2
    data_filtered = data.loc[genestokeep] # select genes to keep from NON NORMALISED df

    #Normalise samples using edgeR TMM (using python package conorm)
    data_tmm = cn.tmm(data_filtered)
    #data_tmm_factors = cn.tmm_norm_factors(data_filtered) #can return norm factors if needed

    #CPM of TMM normalised samples
    nm = norm()
    nm.cpm(df=data_tmm) # CPM of TMM normalised samples
    data_tmm_cpm = nm.cpm_norm

    #Transpose data (samples rows, genes as columns)
    tdata_tmm_cpm = data_tmm_cpm.transpose()

    #Filter samples 
    sample_means = (tdata_tmm_cpm > 0).mean(axis=1)  # mean proportion of expressed genes per sample (axis=1 selects over the row (per sample))
    mean_all = sample_means.mean() # mean of "sample means"
    std_all = sample_means.std() # stdev of "sample means" 

    print("Average proportion of expressed genes (> 0) across all samples (Sample Mean avg.):" , mean_all)
    print("Standard Deviation of expressed genes (> 0) across all samples (Sample Stdev avg.):" , std_all)

    #Calculate user filter: number of SD's away from the mean (filter_sample)
    user_filter = (filter_sample * std_all)
    print("Stdev avg. * user filter :", user_filter)
    print("Range (Sample mean avg. +/- Stdev avg.) for filtering samples to keep: " , (mean_all - user_filter), "to" , (mean_all + user_filter))
    print("Sample means :", sample_means)

    #Caculate if the mean number of expressed genes in any sample is outside range
    #1. Find mean # genes per sample with expression >0, 
    #2. Keep samples with number of expressed genes i: HIGHER than the global mean minus user_filter (# SD's from mean) AND ii) lower than global mean plus user filter
    tokeep = (sample_means >= (mean_all - user_filter)) & (sample_means <= (mean_all + user_filter)) 
    
    #Filter original df, keeping those samples that satisfy reqs
    data_final = tdata_tmm_cpm.loc[tokeep]   # filter original df (already filtered for genes) (with +ve and -ve values), keeping those samples 

    #Return final data
    return data_final