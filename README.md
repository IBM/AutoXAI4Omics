# Auto-Omics: an Explainable Auto-AI tool for omics and tabular data

Auto-Omics is a command line automated explainable AI tool that easily enable researchers to perform phenotype prediction from omics data (e.g., gene expression; microbiome data; or any tabular data) and any tabular data (e.g., clinical) using a range of ML models.

*Key features include*:

* preprocessing specific for omics data (optional)
* feature selection (optional)
* HPO (hyper-parameter optimization) of a variety of ML models including neural networks
* selection of the best ML model(s)
* generation of explainability and interpretability results (using SHAP and Eli5)
* generation of predictive performance scores (cvs files) and a series of visualisations (e.g., plots)
* json configuration file for the specification of preprocessing, ML models, hyperparameter tuning, plots and results produced
* wrapping of Keras and TensorFlow models to make them compatible with sklearn models
* prediction on new data using the best model
* packaged as a Docker container

## Important note

This tool is for IBM internal use ONLY.

## Citation

For general IBM internal use of the tool please cite this article:

* Carrieri, A.P., Haiminen, N., Maudsley-Barton, S. et al. Explainable AI reveals changes in skin microbiome composition linked to phenotypic differences. Sci Rep 11, 4565 (2021). <https://doi.org/10.1038/s41598-021-83922-6>

## Requirements

* Docker
  * installation: `https://docs.docker.com/get-docker/`
* Git
  * installation: `https://github.com/git-guides/install-git`

## How to install Auto-Omics

 1. Clone this repo however you choose (cli command: `git clone --single-branch --branch main git@github.ibm.com:BiomedSciAI-Innersource/Auto-Omics.git`)
 2. Make sure `docker` is running (cli command: `docker version`, if installed the version information will be given)
 3. Within the `Auto-Omics` folder:
       1. Run the following cli command to build the image: `./build.sh -r`
       2. Manually create a new folder called `experiments`

NOTE: if training is run by mistake without first creating the `experiments` directory, and the directory is created while training, the directory needs to be removed and then created again before running training (has to do with access permissions)

## User manual

Everything is controlled through a config dictionary, examples of which can be found in the `configs/exmaples` folder. For an explanation of all parameters, please see the [***CONFIG MANUAL***](https://github.ibm.com/BiomedSciAI-Innersource/Auto-Omics/blob/main/configs/CONFIG_MANUAL.md).

The tool is launched in the cli using `auto_omics.sh` which has multiple flags, examples will be given below:

* `-m` this specifies what mode you want to run auto omics in the options are:
  * `feature` - Run feature selection on a input data set
  * `train` - Tune and train various machine learning models, generate plots and results
  * `test` - To test and evaluate the tuned and trained machine learning models on a completely different holdout dataset
  * `predict` - Use trained models to predict on unseen data
  * `plotting` - If the models have been tuned and trained (and therefore saved), the plots and results can be generated in isolation
  * `bash` - Use to open up a bash shell into the tool
* `-c` this is the filename of the config json within the `Auto-Omics/configs` folder that is going to be given to auto_omics
* `-r` this sets the contain to run as root. Only possibly required if you are running in `bash` mode
* `-g` this specifies if you want auto_omics to use the gpus that are available on the machine (UNDER TESTING)

Data to be used by Auto-Omics needs to be stored in the `Auto-Omics/data` folder.

### Examples

* Run auto omics in training mode with a config called `my_fun_config.json` within the `configs` folder:
  * `./auto_omics.sh -m train -c my_fun_config.json`

* If you wish to run a bash shell within the auto omics image then you can do it using the following. In addition if you wish to be logged in as root add the -r flag:
  * `./auto_omics.sh -m bash -r`

* **UNDER TESTING** If you wish to utilise any gpus that are available on your machine during your auto_omics run then you can add the `-g` flag:
  * `./auto_omics.sh -m train -c my_fun_config.json -g`
