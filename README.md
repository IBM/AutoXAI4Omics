# EAU

The aim of the framework is to easily allow phenotype prediction from omics data or clinical (e.g., gene expression; microbiome datat (currently setup for _biom_ data, handled through `calour`); clinical or any tabular data), using a range of ML models, and various plots (including explainability and feature importance through SHAP and ELI5). 

Key features include:

* Json Config file allows specification of preprocessing, ML models, hyperparameter tuning, plots and results produced
* Wrapping of Keras and TensorFlow models to make them compatible with sklearn models

## Quick Get Started
The general python requirements have been listed below. To install on MacOSX, use the following command:
```
conda env create -f microbiome_osx.yml
```

Everything is controlled through a config dictionary, an example of which can be found in the configs/ folder. For an explanation of all parameters, please see the XXX.

The main way to run the pipeline (to train and hypertune the models) is:
```
python train_models.py -c example_config.json
```

If the models have already been trained (and therefore saved), the plots and results can be generated in isolation by running:
```
python plotting.py -c example_config.json
```

The results are saved in the results/ folder in the subdirectory specified by the `name` field in the JSON config file.

## Requirements
* Python 3.6+
* Calour (which needs scikit-bio)
* Numpy, Scipy, Pandas, Scikit-learn
* Tensorflow 1.12 (at least <2.0), Keras
* Matplotlib, Seaborn
* XGBoost
* SHAP, ELI5
* UMAP (for dimensionality reduction)

## User Guide
For an explanation of the config file, and more detailed information about the framework and how to extend it, see the ***USER GUIDE***
