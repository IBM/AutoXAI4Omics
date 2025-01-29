<!--
 Copyright 2024 IBM Corp.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Config Manual

The JSON config file is at the center of the framework - it controls everything to be run. The [example folder](examples/) contains various examples of config files that can be supplied. Here we shall cover the contents of configs.

Config sections:

- `data` - ***MANDATORY*** - [This section](#data-entry) contains all the information relevant to the data that is to be used in the run
- `ml` - ***MANDATORY*** - [This section](#machine-learning-entry) contains all the relevant information for the machine learning configurations
- `plotting` - ***OPTIONAL*** - [This section](#plotting-entry) contains all the relevant information for the plots that are to be produced, if desired
- `microbiome`/`tabular`/`gene_expression`/`metabolomic` - ***OPTIONAL*** - [This section](#omic-entry) contains the relevant information for the omic pre-processing, if required
- `prediction` - ***OPTIONAL*** - [This section](#prediction-entry) contains the relevant information for the prediction task, if required

## General remarks

- If a value for a parameter in the json file is not provided, the value should `null` or "".
- There are specific pre-processing parameters for `data_type` = { `microbiome`, `gene_expression`, `metabolomic`, `tabular`, `R2G` or `other`}. The for other omic types that have not been mentioned (e.g. `proteomic`), can be run through the tool using `other` but will not invoke any special pre-processing.
- For categorical data, phenotypes are listed in alphabetical order in the results

We refer to two types of input files; Input data files hold your dataset e.g. microbiome/gene expression/metabolomic/tabular data and metadata files hold the target you are trying to predict from the input data

- Input data files for microbiome data are expected to be in the format .biom and the corresponding metadata in format .txt
- Input data files for pre-processing of gene expression data are expected as .csv files with genes and their associated expression measurements in rows and tested samples in columns. Column 1 holds the labels for gene names. Similarly, row 1 will contain sample names.
- Input data files for pre-processing of metabolomic and tabular data are expected as .csv files with measurements in rows and tested samples in columns. Column 1 holds the labels for measurement names e.g. metabolites. Similarly, row 1 will contain sample names.
- All other input data files are expected as .csv files that will pass directly into the ML workflow with no pre-processing. As such they are required to have samples in rows and measurements (or features) in columns. Column 1 holds the labels for sample names. Similarly, row 1 will contain measurement (or feature) names.

## Data Entry

This section is for the information that is to be stored in the `data` heading.

- `name`: The name used to create a directory under which all results, models etc. are saved. This is created under the `"results/"` folder under the `"save_path"` (e.g. `"/experiments/results/"`). The needed sub-directories for the results, models and (if any) graphs are created within this experiment folder.
- `data_type`: `microbiome`, `gene_expression`, `metabolomic`, `tabular`, `other` or `R2G`.
  - `R2G` stands for "Read to Go" meaning that the data set the user is inputting has had all the preprocessing already done and has been split into train & test sets. The data will be in one csv file, with a `set` and `label` column. Where `label` has the label for that sample and `set` contains either `train` or `test`, indicating what set that sample is a part of.
- `file_path`: Name of input data file, e.g. "data/skin_closed_reference.biom" if microbiome data, or "tabular_data.csv" if any tabular data, e.g., gene expression data, in a csv file.
- `metadata_file`: Name of metadata file, the file includes target variable to be predicted, e.g. "data/metadata_skin_microbiome.txt". For pre-processing (gene expression, metabolomic, tabular) this file should have as column 1: header "Sample" with associated sample names that correspond to the sample names in `file_path`
- `target`: Name of the target to predict, e.g. "Age", that is either a column within the `medatata_file` or if `metadata_file` is not provided, e.g. `metadata_file`= "", `target` is the name of a column in the data file specified in `file_path`.

## Machine learning entry

This section is for the information that is to be stored in the `ml` heading.

- `problem_type`: The type of problem, either "classification" or "regression".
- `stratify_by_groups`: "Y" or "N" (default "N"). This allows the user to perform the ML analysis stratifying the samples by groups as specified in the `groups` parameter below. If "Y", samples in the same group will not appear in both training and test datasets.
- `groups`: this is the name of a column in the metadata that represent the groups for the stratification of the samples. For instance, if there are time series samples from the same subjects, `groups` could be "Subject_ID".
- `balancing`: "OVER","UNDER", or "NONE" (default "NONE") if the user chooses to perform class balancing of the data of the training data. This functionality work only for classification tasks and makes sense if there the categories/classes are significantly unbalanced.
- `seed_num`: Provide the seed number to be used. This is given to everything that has a `random_state` argument, as well as being used as the general seed (for `numpy` and `tensorflow`).
- `test_size`: The size of the test data (given to scikit-learn's `train_test_split`), e.g., 0.2 if 20% of the dataset is selected as test set and set aside.
- `hyper_tuning`: The type of hyperparameter tuning to be used, either random search "random" or grid "grid" or `null`. In case of `null` the models will be trained with just one set of parameters. The parameters are defined in `model_params.py` for each method. Grid or random search rely on `scikit-learn` implementations.
- `hyper_budget`: The number of random parameter sets to try if `hyper_tuning` is set to "random". This field is not applicable to "grid" search, therefore can be set to "".
- `model_list`: Specify the models to be used in the analysis (the models are defined in the `model_params.py` file). The current models available for both regression and classification task are the following:
  - "rf", Random Forest
  - "knn", K-Nearest Neighbors
  - "adaboost", Adaboost
  - "autoxgboost", XGBoost with automatic Hyper Parameter Optimization implemented. User can change the default settings in the example config file at `autoxgboost_config`. "Timeout" is in minutes.
  - "autolgbm", LightGBM with automatic Hyper Parameter Optimization implemented. User can change the default settings in the example config file at `autolgbm_config`. "Timeout" is in minutes.
  - "autokeras",  An AutoML system based on Keras for automatic tuning of neural networks available at <https://autokeras.com>. User can change the default settings in the example config file at `autokeras_config`. "time_left_for_this_task" and "per_run_time_limit" are in minutes.
- `scorer_list`: Specify the scoring measures to be used to analyse the models(these are defined in `models.py`).
  - For classification tasks: "acc" (accuracy), "f1" (f1-score), "prec" (precision), "recall"
  - For regression tasks: "mse" (mean squared error), "mean_ae" (mean absolute error), "med_ae" (median absolute error), "rmse" (root mean square error), "mean_ape" (mean absolute percentage error), "r2" (r-squared)
The performance of all models on the train and test sets according to these measures are saved in the "results/" sub-folder of the experiment directory.
- `fit_scorer`: The measure that will be used to select the "best" model from the hyper parameter search. Also used as the scoring method for the plots to be generated. It needs to be one of the scores specified in `scorer_list`.
- `encoding`: For classification tasks, it is the type of encoding to be used for the class. It can be `null` to allow sklearn to deal with it as it needs, or it can be set to "label" (for label encoding) or "onehot" (for one-hot encoding). Note that the neural network models always use one-hot encoding, so if not specified they will handle this themselves. This parameter is rarely used and usually set to `null`.
- `feature_selection` : Define the feature selection to be run for the problem. If NO feature selection is desired remove all entries and set `feature_selection` to `null`.
  - `k` : Valid entries are `"auto"` or and integer `x`. `"auto"` will select an optimum number of features to use, `x` will find the `x` best features.
  - `var_threshold`: If the variance for the column is less than or equal the provided threshold, then the column is removed. Applied before any chosen feature selection method.
  - `auto`: This is a dict containing the parameters for the automated feature selection process.
    - `min_features` : This is the minimum number of features that we wish to be selected.
    - `max_features` :  This is the maximum number of features that we wish to be selected. Can be set to `None` to default to the maximum number of columns.
    - `interval` : The range for the number of features to be tested is generated on a logarithmic scale with the minimum being as defined in `min_features` and the max being the total number of columns. This entry defines the size of the logarithmic increment $10^{ -interval}$
    - `eval_model` : This sets what sklearn estimator that shall be used to train a model to evaluate how good each set of chosen k for the feature selection is
    - `eval_metric` : This is the metric that is used to evaluate the trained evaluation model.
  - `method`: This is a dict containing the parameters to define the feature selection method to be used
    - `name` : This is a string equal to either `RFE` or `SelectKBest` which are the two methods available. Note that `SelectKBest` is significantly quicker.
    - `metric`: This is only used if `SelectKBest` is being used and determines what metric is to be used in the method. valid options include: `f_regression`, `f_classif`, `mutual_info_regression` or `mutual_info_classif`
    - `estimator`: This is only used if `RFE` is being used and determines what estimator is fitted at each stage during the `RFE` process.

## Omic entry

As we defined up in the [data entry section](#data-entry) Auto-Omics can work with 4 data types as defined in `data_type`:

- `microbiome` - [section here](#microbiome-parameters)
- `tabular` - [section here](#tabular-parameters)
- `gene_expression` - [section here](#gene-expression-parameters)
- `metabolomic` - [section here](#metabolomic-parameters)

Which ever entry you give in `data_type` you will need to have a corresponding omic section

### Microbiome parameters

These parameters need to specified only if `data_type`= "microbiome", otherwise they can be set as empty strings "". These need to be given under the `microbiome` heading.

- `collapse_tax`: Allows collapsing the taxonomy to the e.g. genus level "g" or species level "s". Uses the `calour.collapse_taxonomy` function (which collapses the entire taxonomic classification up to the level desired, so even if the genus is the same for two samples, if they have different e.g. order, they will be separate).
- `min_reads` : samples with fewer than this many reads will be removed (default 1000)
- `norm_reads` : samples are re-scaled to this many total reads (default 1000, see below)
- `filter_abundance`: low-abundance features are removed, e.g., OTUs with total count less than X across all samples (default 10, see below)
- `filter_prevalence`: OTUs with low prevalence are removed. The default value is 0.01 which means that features occurring in < 1% of the samples (see below)
- `filter_microbiome_samples`: This can either be a list of dictionaries, or a dictionary, which have different behavior/use-cases. In both, the dictionary key is the column, and the value is a list of values which will be used to filter the samples. If a single dictionary is provided, then each key:value pair is taken in isolation, and for a given column all samples that match any of the values are removed. A list of dictionaries is used when there are one or more multi-column criteria for samples to be removed. Each dictionary in the list is treated in isolation. In the example shown in the config, we want to remove samples where they have "Value1" in "Column2" and either "Value1" or "Value2" in "Column5", then we also want to do the same but for "Value2" in "Column2" with either "Value1" or "Value3" in "Column5". Uses the `calour.filtering.filter_by_metadata`. For example, as specified in "microbiome_example_config.json", all the samples that value "UK" for the metadata "COUNTRY" will be removed from the analysis.
- `remove_classes`: A list of values (class labels) that will be removed from the dataset. Uses the column defined in `target`. Only relevant for classification.
- `merge_classes`: This is a dictionary where the key is the new class and the value is a list of values that will be converted into the key. So `{"X": ["A", "B"]}` will convert all "A" and "B" labels into "X" labels. Uses the column defined in `target`. Only relevant for classification.

The microbial sequence count table and metadata, in biom file format, is loaded into the calour library an open-source python library called calour <http://biocore.github.io/calour/>. The loading process filtered out samples with fewer than `min_reads`=1000 reads (default) and then re-scaled each sample to have its counts sum up to `norm_reads`=1000 (default)  by dividing each feature frequency by the total number of reads in the sample and multiplying by 1000. After loading, the data underwent two rounds of filtering and the remaining features were collapsed at the genus level. For these rounds of pre-processing filtering, was used. The first round of filtering removed low-abundance features, e.g., OTUs with total count less than 10 across all samples (`calour.experiment.filter_abundance(10)`). The second filter removed OTUs with low prevalence, e.g., features occurring in < 1% of the samples (`calour.experiment.filter_prevalence(0.01)`). If the user want to modify any of these parameters can do it by modifying the code directly in the functions `utils.create_microbiome_calourexp()` and `utils.filter_biom()` of the python script utils.py.  

### Gene expression parameters

These parameters need to specified only if `data_type`= "gene_expression". These need to be given under the `gene_expression` heading.

- `expression_type`: Format of gene expression data, choices are 'FPKM', 'RPKM', 'TMM', 'TPM', 'Log2FC', 'COUNTS', 'OTHER'. Note that the different gene expression data types are all filtered as per the selected rules below, however, they have different pre-filtering steps;
  - if you specify “COUNTS” then we convert count data to TMM values before filtering
  - if you specify “FPKM”, “RPKM”, “TPM” or “TMM” these go directly into filtering
  - if you select “Log2FC” or “OTHER” these go directly into filtering but here we expect distributions of values that may include both positive and negative values.
- `filter_sample`: Remove samples if no of genes with coverage is >X std from the mean across all samples (default numerical X=1000000)
- `filter_genes`: Remove genes unless they have a gene expression value over X in Y or more samples (default X=0,Y=1 would be specified in the following format in the json file: ["0","1"])
- `output_file_ge`: Processed output file name (it will be in .csv format)
- `output_metadata`: Processed output metadata file name in .csv format (filtered target data and samples to match those remaining after pre-processing for input into ML)

### Metabolomic parameters

These parameters need to specified only if `data_type`= "metabolomic". These need to be given under the `metabolomic` heading.

- `filter_metabolomic_sample`: Remove samples if no of metabolites with measurements is >X std from the mean across all samples (default numerical X=1000000)
- `filter_measurements`: Remove metabolites unless they have a value over X in Y or more samples (default X=0,Y=1 would be specified in the following format in the json file: ["0","1"])
- `output_file_met`: Processed output file name (it will be in .csv format)
- `output_metadata`: Processed output metadata file name in .csv format (filtered target data and samples to match those remaining after pre-processing for input into ML)

### Tabular parameters

These parameters need to specified only if `data_type`= "tabular". These need to be given under the `tabular` heading.

- `filter_tabular_sample`: Remove samples if no of measures with measurements is >X std from the mean across all samples (default numerical X=1000000)
- `filter_tabular_measurements`: Remove measures unless they have a value over X in Y or more samples (default X=0,Y=1 would be specified in the following format in the json file: ["0","1"])
- `output_file_tab`: Processed output file name (it will be in .csv format)
- `output_metadata`: Processed output metadata file name in .csv format (filtered target data and samples to match those remaining after pre-processing for input into ML)

## Plotting entry

These need to be given in the `plotting` heading.

- `plot_method`: A list of the plots to create (as defined in the `define_plots()` function in `plotting.py`). If this list is empty or `null`, no plots are made. The `plotting.py` script can be run separately if the models have been saved, decoupling model and graph creation but still using the same config file. All the generated plots will be saved in the sub-folder `/graphs`. For each  model in the model List, the tool will generate graphs and/or .csv files summarizing the results and named as `<plot name_<model name>.png` or `<results type>_<model name>.csv`

 Currently possible options are listed below:

- Plots available for classification and regression tasks:
  - "barplot_scorer": Barplot showing a comparison in the performance of the models listed in `model_list` on the test set, or unseen samples. In the sub-folder `results/` one .csv file will be saved, `results/scores__performance_results.csv`, containing the scores specified in `scorer_list`(e.g., MAE and MSE) on the test and training datasets for each model in `model_list`.
  - "boxplot_scorer": Boxplot showing a comparison in performances of the models listed in `model_list` resulting from 5 fold cross validation on the entire dataset.
  - "shap_plots": SHAP explainability plots, i.e., shap summary bar plot and shap summary dot plot for each model in `model_list`, `graphs/top_features_AbsMeanSHAP_Abundance_<data>_<model>.csv`
  - "permut_imp_test": Permutation importance plot showing the list of the top features ranked by importance as computed by eli5 permutation importance algorithm using the test dataset. Note that the model has already been fit.

- Options for explainability and feature importance plots:
  - "top_feats_permImp": Number of top ranked features to be visualized in the permutation importance plots, e.g., 10.
  - "top_feats_shap": Number of top ranked features to be visualized in the SHAP plots, e.g., 20.
  - "explanations_data": Data for which SHAP explanation are required. Options available are "test" for the samples in test set, "exemplars" for the exemplar samples in the test set and "all" for all the samples in the dataset.

- Plot avaliable for classification tasks only:
  - "conf_matrix": The confusion matrix computed on the test set, after the model has being trained and tuned. This plot is generated for each model in `model_list`.
  - "roc_curve": The ROC curves are computed on the test set after model training is completed and generated for each model.

- Plots available for regression tasks only. These plots are generated for each model in `model_list`:
  - "hist_overlapped": Histograms showing the overlap between the distributions of true values and predicted values by a given model. This plot is generated for each model in `model_list`.
  - "joint": joint_plot: Scatter plot showing the correlation between true values and predicted values by a given model. Pearson's correlation is also reported.
  - "joint_dens": joint_plot: Joint density plot showing the correlation between true values and predicted values by a given model. Pearson's correlation is also reported.
  - "corr": correlation_plot: Simple correlation plot between true values and predicted values by a given model. Similar to "joint".

### Explainability config parameters

If 'shap_plots'is in `plot_method` list, the following parameters can be specified, these need to be given in the `plotting` heading.

- `top_feats_shap`: Number of top features to be visualized in the shap summary plots, e.g. `top_feats_shap`=10.
- `explanations_data`: subsets of data for which explanations will be provided. Default is "test". Options are:
  - "test": samples in the test dataset
  - "all": entire set of samples, that is training and test dataset
  - "exemplars": examplar samples in the test sets will be selected and explained

### Feauture importance config parameters

These need to be given in the `plotting` heading.

- `top_feats_permImp`: Number of top features to be visualized in the permutation importance plot, e.g. `tops_feats_permImp`=10.

## Prediction entry

Prediction mode can be used once you have trained a set of model on your a data set. The assumption is that the user is going to feed in the exact same feature set that they gave when the trained their model.

When you want to perform the prediction on a dataset you supply the same config file that you used to train the model but with an extra `prediction` section within the config, as shown belown and lunch it by running `./predict.sh config.json` . Note that when the prediction script is run currently it only predicts using the best model.

```
"prediction":{
        "file_path":"/data/testsets/microbiome/microbiome_500.biom",
        "metadata_file": "/data/testsets/microbiome/microbiome_metadata_500_reg.txt",
        "outfile_name": 'prediction_results'
    }
```

The three entries here are:

- `file_path`: The path to the file you wish to predict on
- `metadata_file`: Optional - the path to the accompanying metadata for the prediction file
- `outfile_name`: Optional, defaults to 'prediction_results', is the name of the csv file the prediction results will be saved to.
