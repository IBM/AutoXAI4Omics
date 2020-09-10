# Explainable AI framework for omics 

The aim of the framework is to easily allow phenotype prediction from omics data (e.g., gene expression; microbiome datat (currently setup for _biom_ data, handled through `calour`); or any tabular data), using a range of ML models, and various plots (including explainability and feature importance through SHAP and ELI5). 

Key features include:

* Json Config file allows specification of preprocessing, ML models, hyperparameter tuning, plots and results produced
* Wrapping of Keras and TensorFlow models to make them compatible with sklearn models

## Docker instructions:

* make sure `docker` is running -- e.g. run `docker version` to get version information
* `./build.sh -r` -- installs required packages
* create a new folder whose name matches the `save_path`, e.g. "experiments"
* `./train_models.sh parameters.json` -- trains models and plots results, using parameter file "configs/parameters.json"
* `./plotting.sh parameters.json` -- plots the already trained models that are listed in "configs/parameters.json"

## Quick Get Started

The general python requirements have been listed below. 

***To install on MacOSX*** you can install the python requirements listed below using conda or pip or you can use the following command:
```
conda env create -f microbiome_osx.yml
```

***To install on Windows 10*** see Windows_installation_instructions.docs:

## Requirements
* Python 3.6+
* Calour (which needs scikit-bio)
* Numpy, Scipy, Pandas, Scikit-learn
* Tensorflow 1.12 (at least <2.0), Keras
* Matplotlib, Seaborn
* XGBoost
* SHAP, ELI5
* UMAP (for dimensionality reduction)


## User guide
Everything is controlled through a config dictionary, an example of which can be found in the configs/ folder. For an explanation of all parameters, please see the ***USER GUIDE***.

The main way to run the pipeline (to train and hypertune the models) is:
```
python train_models.py -c example_config.json
```

If the models have already been trained (and therefore saved), the plots and results can be generated in isolation by running:
```
python plotting.py -c example_config.json
```

The results are saved in the results/ folder in the subdirectory specified by the `name` field in the JSON config file.

For an explanation of the config file, and more detailed information about the framework and how to extend it, see the ***USER GUIDE***
