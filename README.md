# Auto-Omics: an Explainable Auto-AI framework for omics and tabular data

The aim of the framework is to easily allow phenotype prediction from omics data (e.g., gene expression; microbiome data; or any tabular data), using a range of ML models, and various plots (including explainability and feature importance through SHAP and ELI5). 

Key features include:

* Json Config file allows specification of preprocessing, ML models, hyperparameter tuning, plots and results produced
* Wrapping of Keras and TensorFlow models to make them compatible with sklearn models

## Citation:
For general use of the tool please cite this article:
* Carrieri, A.P., Haiminen, N., Maudsley-Barton, S. et al. Explainable AI reveals changes in skin microbiome composition linked to phenotypic differences. Sci Rep 11, 4565 (2021). https://doi.org/10.1038/s41598-021-83922-6

## How to install Auto-Omics
* make sure `docker` is running -- e.g. run `docker version` to get version information
* `./build.sh -r` -- installs all required packages (python and R)
* manually create a new folder whose name matches the `save_path` in configuration files, e.g. "experiments" [NOTE: if training is run by mistake without first creating this directory, and the directory is created while training, the directory needs to be removed and then created again before running training (has to do with access permissions)]

## Requirements
* Docker

## User manual
Everything is controlled through a config dictionary, an example of which can be found in the configs/ folder. For an explanation of all parameters, please see the [***MANUAL***](https://gitlab.stfc.ac.uk/hncdi-software/auto-omics/-/blob/main/MANUAL.md).

There are three possible ways to run the workflow. 

* Tune and train various machine learning models, generate plots and results
```
./train_models.sh example_config.json
```

* If the models have been tuned and trained (and therefore saved), the plots and results can be generated in isolation by running:
```
./plotting.sh example_config.json`
```

* To test and evaluate the tuned and trained machine learning models on a completely different holdout dataset, run:
```
./testing_holdout.sh example_config.json
```

The results are saved in the results/folder in the subdirectory specified by the `name` field in the JSON config file.

For an explanation of the config file, and more detailed information about the framework and how to extend it, see the [***MANUAL***](https://gitlab.stfc.ac.uk/hncdi-software/auto-omics/-/blob/main/MANUAL.md).

