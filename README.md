# Auto-Omics: an Explainable Auto-AI tool for omics and tabular data

Auto-Omics is a command line automated explainable AI tool that easily enable researchers to perform phenotype prediction from omics data (e.g., gene expression; microbiome data; or any tabular data) and any tabular data (e.g., clinical) using a range of ML models. 

*Key features include*:
* preprocessing specific for omics data (optional)
* feature selection (optional)
* HPO (hyper-parameter optimization of a variety of ML models including neural networks
* selection of the best ML model(s)
* generation of explainability and interpretability results (using SHAP and Eli5)
* generation of predictive performance scores (cvs files) and a series of visualisations (e.g., plots)
* json configuration file for the specification of preprocessing, ML models, hyperparameter tuning, plots and results produced
* wrapping of Keras and TensorFlow models to make them compatible with sklearn models
* packaged as a Docker container

## Important note:
This tool is for IBM internal use ONLY.

## Citation:
For general IBM internal use of the tool please cite this article:
* Carrieri, A.P., Haiminen, N., Maudsley-Barton, S. et al. Explainable AI reveals changes in skin microbiome composition linked to phenotypic differences. Sci Rep 11, 4565 (2021). https://doi.org/10.1038/s41598-021-83922-6

## How to install Auto-Omics
* make sure `docker` is running -- e.g. run `docker version` to get version information
* `./build.sh -r` -- installs all required packages (python and R)
* manually create a new folder whose name matches the `save_path` in configuration files, e.g. "experiments" [NOTE: if training is run by mistake without first creating this directory, and the directory is created while training, the directory needs to be removed and then created again before running training (has to do with access permissions)]

## Requirements
* Docker

## User manual
Everything is controlled through a config dictionary, an example of which can be found in the configs/ folder. For an explanation of all parameters, please see the [***MANUAL***](https://github.ibm.com/BiomedSciAI-Innersource/Auto-Omics/blob/main/MANUAL.md).

There are three possible ways to run the workflow. 

* Tune and train various machine learning models, generate plots and results
```
./train_models.sh example_config.json
```

* If the models have been tuned and trained (and therefore saved), the plots and results can be generated in isolation by running:
```
./plotting.sh example_config.json
```

* To test and evaluate the tuned and trained machine learning models on a completely different holdout dataset, run:
```
./testing_holdout.sh example_config.json
```

The results are saved in the results/folder in the subdirectory specified by the `name` field in the JSON config file.

For an explanation of the config file, and more detailed information about the framework and how to extend it, see the [***MANUAL***](https://github.ibm.com/BiomedSciAI-Innersource/Auto-Omics/blob/main/MANUAL.md).

