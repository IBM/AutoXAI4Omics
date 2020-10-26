# User manual
This is a user manual for the framework, where the fields of the config file are explained, and some explanation about how to use and extend the framework is provided.
​
## Config File Explanation
The JSON config file is at the centre of the framework - it controls everything to be run. The `example_config.json` looks like:
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
* If a value is not provided, the value should be provided as `null` or ""
* 

### General parameters
* `name`: The name used to create a directory under which all results, models etc. are saved. This is created under the `"results/"` folder in the main directory. The needed subdirectories for the results, models and (if any) graphs are created within this experiment folder.
* `data_type`: "microbiome" or "gene_expression"
* `file_path`: name of input data file, e.g. "data/skin_closed_reference.biom"
* `metadata_file`: name of metadata file, the file includes target variable to be predicted, e.g. "data/metadata_skin_microbiome.txt"
 * `target`:  within the `medatata_file`, the name of the column with the value to be predicted, e.g. "Age"

### Machine learning parameters
* `class_name`: The name of the column/feature to use as the target (either regression or classification).
* `problem_type`: The type of problem, either "classification" or "regression".
* `merge_classes`: This is a dictionary where the key is the new class and the value is a list of values that will be converted into the key. So `{"X": ["A", "B"]}` will convert all "A" and "B" labels into "X" labels. Uses the column defined in `class_name`. Only relevant for classification.
* `encoding`: The type of encoding to be used for the class. Accepts `null` to allow sklearn to deal with it as it needs, or can specify "label" (for label encoding) or "onehot" (for one-hot encoding). Note that the neural network models always use one-hot encoding, so if not specified they will handle this themselves. This is rarely used.
* `seed_num`: Provide the seed number to be used. This is given to everything that has a `random_state` argument, as well as being used as the general seed (for `numpy` and `tensorflow`).
* `test_size`: The size of the test data (given to scikit-learn's `train_test_split`).
* `hyper_tuning`: The type of hyperparameter tuning to be used, one of: "grid", "random", "boaas", or `null` (which means training with just one set of parameters). The parameters are defined in `model_params.py` for each method. Currently, the _CustomModels_ are not integrated with grid or random search, which rely on `scikit-learn`.
* `hyper_budget`: The number of parameter sets to try. Not applicable to "grid" search.
* `model_list`: Specify the models to be used in the analysis (the models are defined in the `model_params.py` file).
* `scorer_list`: Specify the scoring measures to be used to analyse the models. These are defined in `models.py`. The performance of all models on the train and test sets according to these measures are saved in the "results/" subfolder of the experiment directory.
* `fit_scorer`: The measure that will be used to select the "best" model from the hyperparameter search. Also used as the scoring method for plots.

### Microbiome data pre-processing parameters
* `biom_file`: The name of the relevant .biom file to be loaded in for `calour` (must be in "data" folder)
* `metadata_file`: The name of the metadata file to be loaded in for `calour` (must be in "data" folder)
* `collapse_tax`: Allows collapsing the taxonomy to the e.g. genus level. Uses the `calour.collapse_taxonomy` function (which collapses the entire taxonomic classification up to the level desired, so even if the genus is the same for two samples, if they have different e.g. order, they will be separate).

* `remove_classes`: A list of values (class labels) that will be removed from the dataset. Uses the column defined in `class_name`. Only relevant for classification.
* `filter_samples`: This can either be a list of dictionaries, or a dictionary, which have different behaviour/use-cases. In both, the dictionary key is the column, and the value is a list of values which will be used to filter the samples. If a single dictionary is provided, then each key:value pair is taken in isolation, and for a given column all samples that match any of the values are removed. A list of dictionaries is used when there are one or more multi-column criteria for samples to be removed. Each dictionary in the list is treated in isolation. In the example shown in the config, we want to remove samples where they have "Value1" in "Column2" and either "Value1" or "Value2" in "Column5", then we also want to do the same but for "Value2" in "Column2" with either "Value1" or "Value3" in "Column5".

### Gene expression data pre-processing parameters
* ...
* ...

### Explainability config parameters
* `top_feats_permImp`: ....
* `top_feats_shap`: ....
* `explanations_data`: "...

### Plotting config parameters
 * `plot_method`: A list of the plots to create (as defined in the `define_plots()` function in `plotting.py`). If this list is empty or `null`, no plots are made. The `plotting.py` script can be run separately if the models have been saved, decoupling model and graph creation but still using the same config file.

## Extending the Framework
This section will briefly outline the steps needed to extend the framework. This is mainly aimed at pointing the user towards the relevant dictionaries when a new model or plot is to be added.
​
There are a few parts of this framework that are hard-coded, and will need to be modified when adding new plots, models, or scorers.
​
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
