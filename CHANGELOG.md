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

# CHANGELOG

Change log for the codebase. Initialised from the developments following version `V0.11.3`

## [V1.0.1] - 2024-03-11

### Changed

- cleaned up docstring comment
- removed copyright notices from example json and data
- Readme updated

### Removed

- Removed config jsons that do not have paired data files

### Fixed

- copy-paste corrections
- redundant set casting

### Security

- switched `os.mkdir` for `os.makedirs`

## [V1.0.0] - 2024-01-22

- Introduction of pre-commit hooks (Ruff & Blacks)
- Linting of code
- Creation of feature selection mode for AO
- max_features kwarg added for feature selection
- moved source code into an src folder & modified dockerfile
- created utils subfolder
- created models subfolder
- added in structure.md within ./src to explain the structure format adopted
- renamed mode scripts to have a `mode_` pre-fix
- added extra assertion in mode checks
- removed redundant `check_config`, `check_keys`, `load_params_json`, `activate`
- refactored `plot_graphs` to remove unnessicary `plot_dict` var
- extracted load funcs from utils into own file
- extracted save funcs from utils into own file
- extracted util funcs from `mode_plotting` into respective files
- extracted plotting funcs from `mode_plotting` into own files
- removed exit() calls
- bugfixes
- extracted save funcs from models into utils.save
- extracted load funcs from data_processing into utils.load
- streamlined init
- removed globals() call
- extracted funcs from data_processing into appropriate locations, and file fully dispersed
- changed call of parsing FS entry to correct place
- separated problem specific plots into own files
- extracted shap and permutation importance plots into their own subfolders given their importance
- consolidate metric functions into one location
- consolidate model defs into one location
- custom model clean up
- shifted repeated save_model code from TabAuto children into TabAuto
- shifted repeated load_model code from TabAuto children into TabAuto, except for FixedKeras & AutoKeras being keras based they load in a different was so have left their re-implementation of load model
- shifted repeated fit code from TabAuto children into TabAuto
- shifted repeated init code from TabAuto children into TabAuto
- generalised repeated _define_model code and shifted from TabAuto children into TabAuto
- removed repeated set_params from TabAuto
- removed unnessicary data being passed to plot_reg, plot_clf, perm_imp and most of plot_both functions
- removed unnessicary data being passed to plots_shap functions and separated out problem specific logic into separate function.
- extract repeated getting model_path code to function
- streamlined custom model setup
- TabAuto Class merged with CustomModel Class
- Deleted autogulon src code, as was depreciated ages ago
- Custom SKLearnModel code only ever instansiated in auto mode for AutoSklearn so removed redundant code
- added test coverage config
- tests, type hinting and docstring created for:
  - metrics.define_scorers
  - utils.ml.class_balancing
  - utils.ml.data_split.std_split & strat_split
  - plotting.plot_utils
- reduce ussage of full config dict in:
  - R_replacement.py
- added probabilities for predictions when doing classification problems
- code comment clean up
- multiline strings causing extra space fixed
- added pytest marks to distinguish binary classiification problems
- microbiome pre-processing bugfix
- lgbm warning re: suggest_loguniform & suggest_uniform resolved
- AutoKeras bugfix re: batch_size
- tests_mode grabbing the wrong file at times when lots present
- Arm fixes for previous security fix
- test_modes bugfix
- name change within code base & repo (from `Auto-Omics` to `AutoXAI4Omics`)
- provided example data & config users can run
- corrected test_model_outputs
- added skip conditions for test_omic_datasets
- bug fix of xticks for bar & box plots
- added statement about supported python version to Readme and dev manual
- License file created and headers added to files
- fixed commenting of license in requirements.txt
- `.pre-commit-config.yaml` cleaned
- tests requiring a container marked and fixture added to build the container
- `CONTRIBUTING.md` added and info added to `DEV_MANUAL.md`
- `DEV_MANUAL.md` updated
- Detect secrets added
- Upgraded python base image from `3.9.14` to `3.9.18` for additional security fixes

[V1.0.1]: https://github.com/IBM/AutoXAI4Omics/releases/tag/V1.0.1
[V1.0.0]: https://github.com/IBM/AutoXAI4Omics/releases/tag/V1.0.0
