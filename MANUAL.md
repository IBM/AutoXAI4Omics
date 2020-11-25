# User manual
This is a user manual for the framework, where the fields of the config file are explained, and some explanation about how to use and extend the framework is provided.
​
## Config File Explanation
The JSON config file is at the centre of the framework - it controls everything to be run. The `microbiome_example_config.json` to run the analysis on microbiome data including the pre-processing, looks like:
```{
    "name": "microbiome_smoking",
    "data_type": "microbiome",
    "file_path": "data/skin_closed_reference.biom",
    "metadata_file": "data/metadata_skin_microbiome.txt",
    "target": "SMOKER",
    "seed_num": 42,
    "test_size": 0.2,
    "problem_type": "classification",
    "hyper_tuning": "random",
    "hyper_budget": 100,
    "scorer_list": [
        "f1"
    ],
    "fit_scorer": "f1",
    "model_list": [
        "rf",
        "xgboost",
        "adaboost",
        "svm",
        "knn"
    ],
      "plot_method": [
        "conf_matrix",
        "boxplot_scorer",
        "barplot_scorer",
        "shap_plots",
        "permut_imp_alldata"
      ],
    "top_feats_permImp": 10,
    "top_feats_shap": 10,
    "explanations_data": "test",
    "collapse_tax": "genus",
    "remove_classes": null,
    "merge_classes": null,
    "encoding": null,
    "filter_samples": [
        {
            "COUNTRY": [
                "UK"
            ]
        }
    ]
}
```
### General remarks
* Note that the .json file needs to be in the directory called `configs`.
* If a value for a parameter in the json file is not provided, the value should `null` or "".
* There are specific pre-processing parameters for `data_type` = { `microbiome`, `gene_expression`}. The `data_type` can have any other value or be an empty string (e.g. "metabolomic", "tabular", "", etc.),  but those will not invoke any special pre-processing
* Input files are expected to be .csv or .biom for microbiome data
* Input files for pre-processing of gene expression data are expected as .csv files with genes and their associated expression measurements in rows and tested samples in columns. The cell in column 1, row 1 requires the header "gene" and the rest of this column 1 holds the labels for gene names. Similarly, the remainder of row 1 will contain sample names.

### General parameters
* `name`: The name used to create a directory under which all results, models etc. are saved. This is created under the `"results/"` folder in the main directory. The needed subdirectories for the results, models and (if any) graphs are created within this experiment folder.
* `data_type`: "microbiome" or "gene_expression" or anything else e.g. "metabolomic, but the latter does not currently invoke any specific pre-processing.
* `file_path`: Name of input data file, e.g. "data/skin_closed_reference.biom" if microbiome data, or "tabular_data.csv" if any tabular data, e.g., gene expression data, in a csv file. 
* `metadata_file`: Name of metadata file, the file includes target variable to be predicted, e.g. "data/metadata_skin_microbiome.txt".
 * `target`: Name of the target to predict, e.g. "Age", that is either a column within the `medatata_file` or if `metadata_file` is not provided, e.g. `metadata_file`= "", `target` is the name of a column in the data file specified in `file_path`.

### Machine learning parameters 
* `problem_type`: The type of problem, either "classification" or "regression".
* `seed_num`: Provide the seed number to be used. This is given to everything that has a `random_state` argument, as well as being used as the general seed (for `numpy` and `tensorflow`).
* `test_size`: The size of the test data (given to scikit-learn's `train_test_split`), e.g., 0.2 if 20% of the dataset is selected as test set and set aside.
* `hyper_tuning`: The type of hyperparameter tuning to be used, either random search "random" or grid "grid" or `null`. In case of `null` the models will be trained with just one set of parameters. The parameters are defined in `model_params.py` for each method. Grid or random search rely on `scikit-learn` implementations. 
* `hyper_budget`: The number of random parameter sets to try if `hyper_tuning` is set to "random". This field is not applicable to "grid" search, therefore can be set to "".
* `model_list`: Specify the models to be used in the analysis (the models are defined in the `model_params.py` file). The current models available for both regression and classification task are the following:
    * "xgboost", XGBoost
    * "rf", RandomForest
    * "svm", Support Vector Machines
    * "knn", K-Nearest Neighbors
    * "adaboost", Adaboost
    * "mlp_keras" a Multi Layer Perceptron (MLP) implemented in Keras and defined in custom_models.py
    * "mlp_ens" an ensemble MLP implemented in tensor flow and defined in custom_models.py
Note that `mlp_keras`,  and `mlp_ens`  are not integrated with grid or random search, which rely on scikit-learn. @TODO: to be updated after integration with Panos autoAI models.
* `scorer_list`: Specify the scoring measures to be used to analyse the models(these are defined in `models.py`). 
    * For classification tasks: "acc" (accuracy), "f1" (f1-score), "prec" (precision), "recall"
    * For regression tasks: "mse" (mean squared error), "mean_ae" (mean absolute error), "med_ae" (median absolute error), "rmse" (root mean square error) 
The performance of all models on the train and test sets according to these measures are saved in the "results/" subfolder of the experiment directory.
* `fit_scorer`: The measure that will be used to select the "best" model from the hyperparameter search. Also used as the scoring method for the plots to be generated. It needs to be one of the scores specified in `scorer_list`. 
* `encoding`: For classification tasks, it is the type of encoding to be used for the class. It can be `null` to allow sklearn to deal with it as it needs, or it can be set to "label" (for label encoding) or "onehot" (for one-hot encoding). Note that the neural network models always use one-hot encoding, so if not specified they will handle this themselves. This parameter is rarely used and usually set to `null`. 

### Microbiome data pre-processing parameters
These parameters need to specified only if `data_type`= "microbiome", otherwise they can be set as empty strings ""
* `collapse_tax`: Allows collapsing the taxonomy to the e.g. genus level "g" or species level "s". Uses the `calour.collapse_taxonomy` function (which collapses the entire taxonomic classification up to the level desired, so even if the genus is the same for two samples, if they have different e.g. order, they will be separate).
* `filter_microbiome_samples`: This can either be a list of dictionaries, or a dictionary, which have different behaviour/use-cases. In both, the dictionary key is the column, and the value is a list of values which will be used to filter the samples. If a single dictionary is provided, then each key:value pair is taken in isolation, and for a given column all samples that match any of the values are removed. A list of dictionaries is used when there are one or more multi-column criteria for samples to be removed. Each dictionary in the list is treated in isolation. In the example shown in the config, we want to remove samples where they have "Value1" in "Column2" and either "Value1" or "Value2" in "Column5", then we also want to do the same but for "Value2" in "Column2" with either "Value1" or "Value3" in "Column5". Uses the `calour.filtering.filter_by_metadata`. For example, as specified in "microbiome_example_config.json", all the samples that value "UK" for the metadata "COUNTRY" will be removed from the analysis. 
* `remove_classes`: A list of values (class labels) that will be removed from the dataset. Uses the column defined in `target`. Only relevant for classification.
* `merge_classes`: This is a dictionary where the key is the new class and the value is a list of values that will be converted into the key. So `{"X": ["A", "B"]}` will convert all "A" and "B" labels into "X" labels. Uses the column defined in `target`. Only relevant for classification.

### Gene expression data pre-processing parameters
Examples of usage of these parameters is available in file: gene_exp_regression.json
* `expression_type`: Format of gene expression data, choices are 'FPKM', 'RPKM', 'TMM', 'TPM', 'Log2FC', 'COUNTS', 'OTHER'. Note that the different gene expression data types are all filtered as per the selected rules below, however, they have different pre-filtering steps;
    * if you specify “COUNTS” then we convert count data to TMM values before filtering
    * if you specify “FPKM”, “RPKM”, “TPM” or “TMM” these go directly into filtering 
    * if you select “Log2FC” or “OTHER” these go directly into filtering but here we expect distributions of values that may include both positive and negative values. 
* `filter_sample`: Remove samples if no of genes with coverage is >X std from the mean across all samples, default numerical X=1000000 
* `filter_genes`: Remove genes unless they have a gene expression value over X in Y or more samples (default X=0,Y=1 would be specified in the following format in the json file: ["0","1"])
* `output_file_ge`: Processed output file name (it will be in .csv format)

### Plotting config parameters
 * `plot_method`: A list of the plots to create (as defined in the `define_plots()` function in `plotting.py`). If this list is empty or `null`, no plots are made. The `plotting.py` script can be run separately if the models have been saved, decoupling model and graph creation but still using the same config file. All the generated plots will be saved in the sub-folder `/graphs`. For each  model in the model List, the tool will generate graphs and/or .csv files summarising the results and named as `<plot name_<model name>.png` or `<results type>_<model name>.csv`
 
 Currently possible options are listed below:

* Plots available for classification and regression tasks:
    * "barplot_scorer": Barplot showing a comparison in the performance of the models listed in `model_list` on the test set, or unseen samples. In the sub-folder `results/` one .csv file will be saved, `results/scores__performance_results.csv`, containing the scores specified in `scorer_list`(e.g., MAE and MSE) on the test and training datasets for each model in `model_list`.
    * "boxplot_scorer": Boxplot showing a comparison in performances of the models listed in `model_list` resulting from 5 fold cross validation on the entire dataset. 
    * "shap_plots": SHAP explainability plots, i.e., shap summary bar plot and shap summary dot plot for each model in `model_list`, `graphs/top_features_AbsMeanSHAP_Abundance_<data>_<model>.csv`
    * "permut_imp_all_data_cv": Permutation importance plot showing the list of the top feautures ranked by importance as computed by eli5 while performing 5 cross validation using the entire dataset. For each model the scores of 5CV are saved in `results/<scores_5CV_<model>.csv>`

* Options for explainability and feature importance plots:
    * "top_feats_permImp": Number of top ranked features to be visualised in the permutation importance plots, e.g., 10. 
    * "top_feats_shap": Number of top ranked features to be visualised in the SHAP plots, e.g., 20.
    * "explanations_data": Data for which SHAP explanation are required. Options available are "test" for the samples in test set, "exemplars" for the examplar samples in the test set and "all" for all the samples in the dataset. 
    
* Plot avaliable for classification tasks only:
    * "conf_matrix": The confusion matrix computed on the test set, after the model has being trained and tuned. This plot is generated for each model in `model_list`.
    
* Plots available for regression tasks only. These plots are generated for each model `in model_list`: 
    "hist_overlapped": Histograms showing the overlap between the distributions of true values and predicted values by a given model. This plot is generated for each model in `model_list`. 
    "joint": joint_plot: Scatter plot showing the correlation between true values and predicted values by a given model. Pearson's correlation is also reported.
    "joint_dens": joint_plot: Joint density plot showing the correlation between true values and predicted values by a given model. Pearson's correlation is also reported.
    "corr": correlation_plot: Simple correlation plot between true values and predicted values by a given model. Similar to "joint". 
    
### Explainability config parameters
If 'shap_plots'is in `plot_method` list, the following parameters can be specified:
* `top_feats_shap`: Number of top features to be visualised in the shap summary plots, e.g. `top_feats_shap`=10.
* `explanations_data`: subsets of data for which explanations will be provided. Default is "test". Options are:
    * "test": samples in the test dataset
    * "all": entire set of samples, that is training and test dataset
    * "exemplars": examplar samples in the test sets will be selected and explained

### Feauture importance config parameters
* `top_feats_permImp`: Number of top features to be visualised in the permutation importance plot, e.g. `tops_feats_permImp`=10.

## Extending the Framework
This section will briefly outline the steps needed to extend the framework. This is mainly aimed at pointing the user towards the relevant dictionaries when a new model or plot is to be added.
​
There are a few parts of this framework that are hard-coded, and will need to be modified when adding new plots, models, or scorers.
​
### Adding a new data type
A new `data_type` option can be added to the code, with associated specific pre-procrssing steps.

### Adding a new plot
In `plotting.py`, the `define_plots()` function at the top specifies which plotting functions are available for regression and classification problems. Some plots are applicable to both, so add the alias (which is used in the config file) and the function object itself to the relevant dictionary (or -ies).
​
The function itself then needs to be added to the `plot_graphs()` function with the relevant arguments. Some functions have been duplicated here with different arguments for easy access via the alias (allowing multiple calls to the same function from a single config file call).
​
For plots that load a Tensorflow or Keras model, after that model is used you will need to call `utils.tidy_tf()` to ensure that there is no lingering session or graph. This is called after every plot function, but when loading multiple Tensorflow models this will need to be called inside the plotting function.
​
All plotting functions have a save argument to allow plots to be shown on the screen or saved, though this defaults to `True`. For uniform parameters, when saving use the `save_fig()` function that calls the usual `fig.savefig` function in matplotlib. When loading models, do this through the `utils.load_model()` function. For defining the saving and loading for a _CustomModel_, see the section below about adding models.
​
If the model has a useful hook to SHAP e.g. via the _TreeExplainer_, then make sure it is added in `utils.select_explainer()`.
​
There is a `plotting.pretty_names()` function that specifies some better looking names for the aliases of the scorers and models. Either for new models/scorers or existing ones, change the values there to affect the text on plots.
​
### Adding a new model
To add a new model, the parameter definitions need to added to `model_params.py`, which has separate dictionaries for parameter definitions for grid or random search, as well as a single model. Similarly, new models need to be added to `models.define_models()`, which includes any specific modifications for the parameter definitions for classification or regression. If larger changes are needed, you should add a separate model specifically for the problem type.
​
#### CustomModel
In addition to the above, if the model is not part of scikit-learn, then it can be added as a subclass of the _CustomModel_ class (in `custom_model.py`). The methods of the base class show what needs to be defined in order for it to behave similarly to a sklearn model.
​
The key things to keep in mind are a way to save and load models, which may require temporarily deleting attributes that cannot be pickled e.g. a Tensorflow graph. Thus, when loading, these attributes will need to be added back in, e.g. by defining the graph again. If you encounter errors, first look at how the other subclasses (`MLPEnsemble` wrapping Tensorflow, and `MLPKeras` wrapping Keras) handled it.
​
Each subclass should has a `nickname` class attribute, which is the model's alias used in the config files. This is automatically taken and stored in `CustomModel.custom_aliases`, which is then used throughout when these models need to be handled differently from the normal sklearn models.
​
### Adding a new scorer
To add a new measure, simply register the function in the dictionary in `models.define_scorers()`. 
​
The only caveat here is that the sklearn convention is that a higher value is better. This convention is used in the hyperparameter tuning, and so when specifying a loss or an error, then when calling `make_scorer()` then you need to pass `greater_is_better=False`. In this case, the values become negative, so when plotting the absolute value needs to be taken (this can also be done for the .csv results if desired, but is not currently).
