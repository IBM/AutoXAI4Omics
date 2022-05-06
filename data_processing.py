import subprocess
import utils
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import scipy.sparse
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_regression, f_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import models
import math


################### STANDARDIS DATA ###################
def standardize_data(data):
    """ 
    Standardize the input X using Standard Scaler
    """

    if scipy.sparse.issparse(data):
        data = data.todense()
    else:
        data = data
        
    SS = StandardScaler()
    data = SS.fit_transform(data)
    return data, SS

def transform_data(data,transformer):
    
    if scipy.sparse.issparse(data):
        data = data.todense()
    else:
        data = data
        
    try:
        data = transformer.transform(data)
        return data
    except:
        raise TypeError("Supplied transformer does not have the transform method")
        
################### LOAD DATA ###################
def get_data(path_file, target, metadata_path):
    """ 
    Read the input files and return X, y (target) and the feature_names
    """

    # Read the data
    data = pd.read_csv(path_file, index_col=0)
    print("Data dimension: "+str(data.shape))

    # Check if the target is in a separate file or in the same data
    if(metadata_path == ""):
        y = data[target].values
        data_notarget = data.drop(target, axis=1)

    else: # it assumes the data does not contain the target column
        # Read the metadata file
        metadata = pd.read_csv(metadata_path, index_col=0)
        y = metadata[target].values
        data_notarget = data

    features_names = data_notarget.columns
    x = data_notarget.values

    # Check the data and labels are the right size
    assert len(x) == len(y)

    return x,y,features_names

def get_data_microbiome(path_file, metadata_path, config_dict):
    '''
    Load and process the data
    '''

    # Use calour to create an experiment
    print("Path file: " +path_file)
    print("Metadata file: " +metadata_path)
    if(config_dict["norm_reads"] == "None" and config_dict["min_reads"] == "None"):
        amp_exp = utils.create_microbiome_calourexp(path_file, metadata_path, None, None)
    else:
        amp_exp = utils.create_microbiome_calourexp(path_file, metadata_path, config_dict["norm_reads"],
                                                    config_dict["min_reads"])
    print("")
    print("")
    print("")
    print("***** Preprocessing microbiome data *******")

    print(f"Original data dimension: {amp_exp.data.shape}")
    # Use calour to filter the data

    amp_exp = utils.filter_biom(amp_exp, collapse_tax=config_dict["collapse_tax"])
    print(f"After filtering contaminant, collapsing at genus and filtering by abundance: {amp_exp.data.shape}")

    # Filter any data that needs it
    if config_dict["filter_samples"] is not None:
        amp_exp = utils.filter_samples(amp_exp, config_dict["filter_samples"])

    # Modify the classes if need be
    amp_exp = utils.modify_classes(
        amp_exp,
        config_dict["target"],
        remove_class=config_dict["remove_classes"],
        merge_by=config_dict["merge_classes"]
    )

    print(f"After filtering samples: {amp_exp.data.shape}")

    print("Save experiment after filtering with name exp_filtered")
    amp_exp.save('biom_data_filtered'+config_dict["name"])
    print("****************************************************")
    print("")
    print("")
    print("")

    # Prepare data (load and normalize)
    x = utils.prepare_data(amp_exp)
    print(x.shape)
    #print(amp_exp.sample_metadata.shape)
    #print(amp_exp.sample_metadata.columns)

    #try:
    # Select the labels
    y = utils.select_class_col(
        amp_exp,
        encoding=config_dict["encoding"], #from Cameron
        name=config_dict["target"]
    )
    #except:
    #   print("!!! ERROR: PLEASE SELECT TARGET TO PREDICT FROM METADATA FILE !!!")

    features_names = utils.get_feature_names_calourexp(amp_exp, config_dict)

    # Check the data and labels are the right size
    assert len(x) == len(y)

    return  x, y, features_names

def get_data_gene_expression(path_file, metadata_path, config_dict):
    '''
    Load and process the data
    '''

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

    python_command = "python AoT_gene_expression_pre_processing.py "+strcommand
    print(python_command)
    subprocess.call(python_command, shell=True)
    x,y, feature_names = get_data(config_dict["output_file_ge"], config_dict["target"], metout_file)

    return x,y, feature_names

def get_data_metabolomic(path_file, metadata_path, config_dict):
    '''
    Load and process the data
    '''

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

    python_command = "python AoT_gene_expression_pre_processing.py "+strcommand
    print(python_command)
    subprocess.call(python_command, shell=True)
    x,y, feature_names = get_data(config_dict["output_file_met"], config_dict["target"], metout_file)

    return x,y, feature_names

def get_data_tabular(path_file, metadata_path, config_dict):
    '''
    Load and process the data
    '''

    # Use calour to create an experiment
    print("Path file: " +path_file)
    print("Metadata file: " +metadata_path)

    print("")
    print("")
    print("")
    print("***** Preprocessing tabular data *******")

    # add the input file parameter
    strcommand = "--expressionfile "+ path_file + " "

    # add the expression type parameter that is required
    expression_type = "TAB"
    strcommand = strcommand+"--expressiontype "+expression_type+ " "

    # add the filter_samples parameter that is optional
    if config_dict["filter_tabular_sample"] is not None:
        filter_tabular_samples = config_dict["filter_tabular_sample"]
        print(filter_tabular_samples)
        strcommand = strcommand + "--Filtersamples " + str(filter_tabular_samples) + " "

    # add the filter_genes parameter to filter measurements that is optional
    if config_dict["filter_tabular_measurements"] is not None:
        filter_measurements = config_dict["filter_tabular_measurements"][0] +" "+config_dict["filter_tabular_measurements"][1]
        print(filter_measurements)
        strcommand = strcommand+"--Filtergenes "+filter_measurements+" "

    # add the output file name that is required
    if config_dict["output_file_tab"] is not None:
        output_file_tab = config_dict["output_file_tab"]
        print(output_file_tab)
        strcommand = strcommand+"--output "+output_file_tab+" "

    # add the metadata for filtering
    strcommand = strcommand+"--metadatafile "+ metadata_path + " "
    
    #add metadata output file that is required
    if config_dict["output_metadata"] is not None:
        metout_file = config_dict["output_metadata"]
        print(metout_file)
        strcommand = strcommand+"--outputmetadata "+metout_file+" " 
    
    print(strcommand)

    python_command = "python AoT_gene_expression_pre_processing.py "+strcommand
    print(python_command)
    subprocess.call(python_command, shell=True)
    x,y, feature_names = get_data(config_dict["output_file_tab"], config_dict["target"], metout_file)

    return x,y, feature_names

def load_data(config_dict,load_holdout=False):
    """
    Load the data presented in the config file
    """
    
    if(config_dict["data_type"]=="microbiome"):
        # This reads and preprocesses microbiome data using calour library -- it would be better to change this preprocessing so that it is not dependent from calour
        x,y,features_names = get_data_microbiome(config_dict["file_path"], config_dict["metadata_file"], config_dict)
    elif(config_dict["data_type"] == "gene_expression"):
        # This reads and preprocesses microbiome data using calour library -- it would be better to change this preprocessing so that it is not dependent from calour
        x, y, features_names = get_data_gene_expression(config_dict["file_path"], config_dict["metadata_file"], config_dict)
    elif(config_dict["data_type"] == "metabolomic"):
        x, y, features_names = get_data_metabolomic(config_dict["file_path"], config_dict["metadata_file"], config_dict)
    elif(config_dict["data_type"] == "tabular"):
        x, y, features_names = get_data_tabular(config_dict["file_path"], config_dict["metadata_file"], config_dict)
    else:
        # At the moment for all the other data types, for example metabolomics, we have not implemented preprocessing except for standardisation with StandardScaler()
        x, y, features_names = get_data(config_dict["file_path"], config_dict["target"], config_dict["metadata_file"])
        
    if load_holdout:
        if(config_dict["data_type"]=="microbiome"):
            # This reads and preprocesses microbiome data using calour library -- it would be better to change this preprocessing so that it is not dependent from calour
            x_heldout, y_heldout, features_names = get_data_microbiome(config_dict["file_path_holdout_data"], config_dict["metadata_file_holdout_data"], config_dict)
        elif(config_dict["data_type"] == "gene_expression"):
            # This reads and preprocesses microbiome data using calour library -- it would be better to change this preprocessing so that it is not dependent from calour
            x_heldout, y_heldout, features_names = get_data_gene_expression(config_dict["file_path_holdout_data"], config_dict["metadata_file_holdout_data"], config_dict)
        elif(config_dict["data_type"] == "metabolomic"):
            x_heldout, y_heldout, features_names = get_data_metabolomic(config_dict["file_path_holdout_data"], config_dict["metadata_file_holdout_data"], config_dict)
        elif(config_dict["data_type"] == "tabular"):
            x_heldout, y_heldout, features_names = get_data_tabular(config_dict["file_path_holdout_data"], config_dict["metadata_file_holdout_data"], config_dict)
        else:
            # At the moment for all the other data types, for example metabolomics, we have not implemented preprocessing except for standardisation with StandardScaler()
            x_heldout, y_heldout, features_names = get_data(config_dict["file_path_holdout_data"], config_dict["target"], config_dict["metadata_file_holdout_data"])
        
    if load_holdout:
        return x, y, x_heldout, y_heldout, features_names 
    else:
        return x, y, features_names

################### SPLIT DATA ###################

def split_data(x, y, config_dict):
    """
    Split the data according to the config (i.e normal split or stratify by groups)
    """
    # Split the data in train and test
    if config_dict["stratify_by_groups"] == "Y":

        x_train, x_test, y_train, y_test = strat_split(x, y, config_dict)
        
    else:
        x_train, x_test, y_train, y_test = std_split(x, y, config_dict)
        
    return x_train, x_test, y_train, y_test

def strat_split(x, y, config_dict):
    """
    split the data according to stratification
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=config_dict["test_size"], random_state=config_dict["seed_num"])
    #gss = GroupKFold(n_splits=7)

    metadata = pd.read_csv(config_dict["metadata_file"], index_col=0)
    le = LabelEncoder()
    groups = le.fit_transform(metadata[config_dict["groups"]])

    for train_idx, test_idx in gss.split(x, y, groups):
        x_train, x_test, y_train, y_test = x[train_idx], x[test_idx], y[train_idx], y[test_idx]
    
    return x_train, x_test, y_train, y_test

def std_split(x, y, config_dict):
    '''
    Determine the type of train test split to use on the data.
    '''
    test_size = config_dict["test_size"]
    seed_num = config_dict["seed_num"]
    problem_type = config_dict["problem_type"]
    
    
    if problem_type == "classification":
        try:
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=test_size,
                    random_state=seed_num, stratify=y
                )
        except:
            print('!!! ERROR: PLEASE SELECT VALID PREDICTION TASK AND TARGET !!!')
            raise

    # Don't stratify for regression (sklearn can't currently handle it with e.g. binning)
    elif problem_type == "regression":
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size,
                random_state=seed_num
            )
        except:
            print('!!! ERROR: PLEASE SELECT VALID PREDICTION TASK AND TARGET !!!')
            raise

    return x_train, x_test, y_train, y_test

################### FEATURE SELECTION ###################
# Potential improvements:
# ability to choose feature selection metric
# ability to choose selection model used 
# ability to choose model evaluation metric
# ability to choose feature selection method
# ability to set min_features & interval
# chain with varinence FS
# gen&save log-acc graph
# gen&save eval-scatter graph

def manual_feat_selection(x,y,k_select,problem_type):
    '''
    Given trainging data this will select the k best features for predicting the target. we assume data has been split into test-train and standardised
    '''
    
    #select correct metric based on problem type
    if problem_type == 'classification':
        # metric = chi2
        metric = f_classif
    elif problem_type == 'regression':
        metric = f_regression
    else:
        raise ValueError("PROBLEM TYPE IS NOT CLASSIFICATION OR REGRESSION")
    
    #perform selection
    SKB = SelectKBest(metric, k=k_select)
    x_trans = SKB.fit_transform(x, y)
    
    return x_trans, SKB

def train_eval_feat_selection_model(x,y,n_feature,problem_type):
    """
    Train and score a model if it were to only use n_feature
    """
    
    x_trans, SKB = manual_feat_selection(x,y,n_feature,problem_type)    #select the best k features
    
    # set the training model we are going to build based on the type of problem
    if problem_type == 'classification':
        selection_model = RandomForestClassifier
        eval_metric = accuracy_score
    elif problem_type == 'regression':
        selection_model = RandomForestRegressor
        eval_metric = mean_squared_error
    else:
        raise ValueError("PROBLEM TYPE IS NOT CLASSIFICATION OR REGRESSION")
    
    # init the model
    fs_model = selection_model(bootstrap=True, max_depth=None, max_features='auto', 
                               max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=3, min_weight_fraction_leaf=0.0, 
                               n_estimators=10, n_jobs=-1,oob_score=False, random_state=42, verbose=0, warm_start=False)
    # fit, predict, score
    fs_model.fit(x_trans,y)
    y_pred = fs_model.predict(x_trans)
    eval_score = eval_metric(y, y_pred)
    
    return eval_score

def k_selector(acc,top=True):
    """
    Given a set of accuracy results will choose the lowest scoring, stable k
    """
    acc = dict(sorted(acc.items()))     #ensure keys are sorted low-high
    sr = pd.DataFrame(pd.Series(acc))    #turn results dict into a series
    
    sr['r_m'] = sr[0].rolling(3,center=True).mean()    #compute the rolling average, based on a window of 3
    sr['r_std'] = sr[0].rolling(3,center=True).std()    #compute the rolling std, based on a window of 3
    
    sr['r_m'].iloc[0] = sr[0].iloc[[0,1]].mean()    #fill in the start values
    sr['r_std'].iloc[0] = sr[0].iloc[[0,1]].std()
    
    sr['r_m'].iloc[-1] = sr[0].iloc[[-2,-1]].mean()  #fill in the end values
    sr['r_std'].iloc[-1] = sr[0].iloc[[-2,-1]].std()
    
    if all(sr[['r_m','r_std']].std() == 0):
        print("CONSISTENT RESULTS - picking lightweight model")
        k_selected = sr.index[0]
    else:
        sr_n = ((sr[['r_m','r_std']]-sr[['r_m','r_std']].mean())/sr[['r_m','r_std']].std())    #normalise the values

        sr_n = sr_n-math.floor(sr_n.min().min())    #adjust the values as we are look for the lowest left hand quadrant
        sr_r = (sr_n['r_m']**2 + sr_n['r_std']**2 )**0.5    #compute the norms
        
        ind_selected = np.where(sr_r==sr_r.min())[0].max() #select the smallest norm, if there are multiple select that with the largest number of features.
        
        #averages were calculated from a window if top is true it will fetch the top k in this winning window
        if top:
            ind_selected +=1 if ind_selected < len(sr_r.index) else 0
        
        #get k value
        k_selected = sr_r.index[ind_selected]
    
    return k_selected

def auto_feat_selection(x,y,problem_type,min_features=10,interval=1):
    """
    Given data this will automatically find the best number of features, we assume the data provided has already been split into test-train and standardised.
    """
    
    print("Generating logarithmic selection for k")
    max_features = x.shape[1]
    #if the max number of features is infact smaller than min_features, set min_features to 1
    if max_features < min_features:
        print("Min features more than given number of features, setting min_features to 1.")
        min_features = 1 
    
    # get a logarithmic spaced selction of potential k
    n_feature_candicates = (10**np.arange(np.log10(min_features),np.log10(max_features)//(10**-interval)/(10**interval),(10**-interval))//1).astype(int).tolist() 
    
    #init result dict
    acc={}
    
    # train and evaluate a model for each potential k 
    for n_feature in n_feature_candicates:
        print(f"Evaluating basic model trained on {n_feature} features")
        acc[n_feature] = train_eval_feat_selection_model(x,y,n_feature,problem_type)
    
    print("Selecting optimum k")
    chosen_k = k_selector(acc) # select the 'best' k based on the results we have attained
    
    print("transforming data based on optimum k")
    #get the transformed dataset and the transformer
    x_trans, SKB = manual_feat_selection(x,y,chosen_k,problem_type)
    
    return x_trans, SKB

def feat_selection(x,y,problem_type,k):
    """
    A function to activate manual or auto feature selection
    """
    
    if k=='auto':
        print("Beginning Automatic feature selection")
        x_trans, SKB = auto_feat_selection(x,y,problem_type,min_features=10,interval=1)
    elif type(k)==int:
        print("Beginning feature selection with given k")
        x_trans, SKB = manual_feat_selection(x,y,k,problem_type)
    else:
        raise ValueError("k must either be an int or the string 'auto' ")
    
    return x_trans, SKB