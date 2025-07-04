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

# Automated Explainable AI for Omics (AutoXAI4Omics): an Explainable Auto-AI tool for omics and tabular data

AutoXAI4Omics is a command line automated explainable AI tool that easily enable researchers to perform phenotype prediction from omics data (e.g., gene expression; microbiome data; or any tabular data) and any tabular data (e.g., clinical) using a range of ML models.

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

## Requirements

* Docker
  * installation: `https://docs.docker.com/get-docker/`
* Git
  * installation: `https://github.com/git-guides/install-git`
* Python 3.9 (only required if the user is planning on contributing to the development of the tool)

## How to install AutoXAI4Omics

 1. Clone this repo however you choose (cli command: `git clone --single-branch --branch main git@github.com:IBM/AutoXAI4Omics.git`)
 2. Make sure `docker` is running (cli command: `docker version`, if installed the version information will be given)
 3. Within the `AutoXAI4Omics` folder:
       1. Run the following cli command to build the image: `./build.sh -r`
       2. Manually create a new folder called `experiments`. IMPORTANT NOTE: if training is run by mistake without first creating the `experiments` directory, and the directory is created while training, the directory needs to be removed and then created again before running training (has to do with access permissions).
       3. Make the experiments folder accessaible by running the following the directory where the experiment directory exists:

       ```shell
       chmod 777 -R experiments 
       ```

## Citation

For citation of this tool, please reference this article:

* James Strudwick, Laura-Jayne Gardiner, Kate Denning-James, Niina Haiminen, Ashley Evans, Jennifer Kelly, Matthew Madgwick, Filippo Utro, Ed Seabolt, Christopher Gibson, Bharat Bedi, Daniel Clayton, Ciaron Howell, Laxmi Parida, Anna Paola Carrieri. doi: <https://doi.org/10.1093/bib/bbae593>
<!-- bioRxiv 2024.03.25.586460;  -->

**NOTE** The configs, data and results published with the paper were produced using version `1.0.0` of the tool. If you wish to reproduced the results please make sure you pull the correct version. Otherwise you will need to update the configs to account for the improvements that have been made in subsequent versions since the initial release.

## User manual

Everything is controlled through a config dictionary, examples of which can be found in the `configs/exmaples` folder. For an explanation of all parameters, please see the [***CONFIG MANUAL***](https://github.com/IBM/AutoXAI4Omics/blob/main/DEV_MANUAL.md).

The tool is launched in the cli using `autoxai4omics.sh` which has multiple flags, examples will be given below:

* `-m` this specifies what mode you want to run AutoXAI4Omics in the options are:
  * `feature` - Run feature selection on a input data set
  * `train` - Tune and train various machine learning models, generate plots and results
  * `test` - To test and evaluate the tuned and trained machine learning models on a completely different holdout dataset
  * `predict` - Use trained models to predict on unseen data
  * `plotting` - If the models have been tuned and trained (and therefore saved), the plots and results can be generated in isolation
  * `bash` - Use to open up a bash shell into the tool
* `-c` this is the filename of the config json or subfolder within the `AutoXAI4Omics/configs` folder that is going to be given to AutoXAI4Omics. If it is a filename `AutoXAI4Omics` will run for that single config. If it is a subfolder within `AutoXAI4Omics` it will enter into batch mode and run all of the config in the provided folder, and any further subfolders, sequentially.
* `-r` this sets the contain to run as root. Only possibly required if you are running in `bash` mode
* `-d` this detatches the cli running the container in the background
* `-n` if you decide to run AutoXAI4Omics in batch mode you can set the maximium number of runs that will run in parallel at the same time, default is 1 (run sequantially with no parallism). It is up to the user to detmine how many parallel runs their system can handle.
* `-g` this specifies if you want AutoXAI4Omics to use the gpus that are available on the machine (UNDER TESTING)

Data to be used by AutoXAI4Omics needs to be stored in the `AutoXAI4Omics/data` folder.

### Examples

* Run AutoXAI4Omics in training mode with a config called `my_fun_config.json` within the `configs` folder:
  * `./autoxai4omics.sh -m train -c my_fun_config.json`
* Run AutoXAI4Omics in training mode with all the configs in the `my_experiments` folder within the `configs` folder:
  * `./autoxai4omics.sh -m train -c my_experiments`
* If you have further nested subfolder's within `my_experiments` folder, such as `my_specific_experiments` you can run those by providing the relative path:
  * `./autoxai4omics.sh -m train -c my_experiments/my_specific_experiments`

* We have provided and example config and dataset that you can run to get going. The components are:
  * config: `configs/examples/50k_barley_SHAP.json`
  * data: `data/geno_row_type_BRIDGE_50k_w.hetero.csv`
  * metadata: `data/row_type_BRIDGE_pheno_50k_metadata_w.hetero.csv`
  * cli command to run: `./autoxai4omics.sh -m train -c examples/50k_barley_SHAP.json`

* If you wish to run a bash shell within the AutoXAI4Omics image then you can do it using the following. In addition if you wish to be logged in as root add the `-r` flag:
  * `./autoxai4omics.sh -m bash -r`

* AutoXAI4Omics has a config duplication function (for when you wish to run the same config over multiple datasets). To use this you need to build the image and the run it in `bash` mode. Once there you can then run:
  
  ```shell
  python mode_config_duplicate.py -c SUBPATH_TO_TEMPLATE_CONFIG -d DATA_SUBDIR
  ```

  where:
  * `SUBPATH_TO_TEMPLATE_CONFIG` is a path within the config dir to a config file to use as the teamplate. e.g. `processed/processed_template.json`
  * `DATA_SUBDIR` is a folder in the data dir containing all the datafiles that the config is to be duplicated e.g. `processed`
    * **NOTE** The duplication currently assumes that the data sets are of only 1 file (i.e no metadata file)
    * **NOTE** The duplicated output configs will be outputted to a dir called `replicates` in the same directory as the original template

* **UNDER DEVELOPMENT** If you wish to utilise any gpus that are available on your machine during your AutoXAI4Omics run then you can add the `-g` flag:
  * `./autoxai4omics.sh -m train -c my_fun_config.json -g`
