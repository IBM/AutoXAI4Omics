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

# Extending the Framework

This section will briefly outline the steps needed to extend the framework. This is mainly aimed at pointing the user towards the relevant dictionaries when a new model or plot is to be added.
​
There are a few parts of this framework that are hard-coded, and will need to be modified when adding new plots, models, or scorers.

Currently the tool uses `python3.9`. As the tool expands we will check and expand the range of python versions that it could be changed/upgraded to.

## Code formatting

Please use blacks & ruff to format any code contributions, we have a pre-commit config yaml that you can use.

## Virtual enviroment

Currently we are unable to make a virtual enviroment on M1 macs (`darwin/arm64`) as  but for `linux` systems a virtual enviroment can be made with the requirements given in `requirements.txt`

## Testing

We have pytests that can be excuted to make sure the system works. The low level tests need to be expanded to increase coverage. The high-level system tests are covered by the `test_modes.py` file. This test covers:

- Where it will build the container and then run it with synthetic data through all the separate modes.
- Comparing the output from the trained run with stored results to ensure reproducibility
- If available, will also omic datasets to ensure it works for these problems too

## Adding a new data type

A new `data_type` option can be added to the code, with associated specific pre-procrssing steps.

## Adding a new plot

In `plotting.py`, the `define_plots()` function at the top specifies which plotting functions are available for regression and classification problems. Some plots are applicable to both, so add the alias (which is used in the config file) and the function object itself to the relevant dictionary (or -ies).
​
The function itself then needs to be added to the `plot_graphs()` function with the relevant arguments. Some functions have been duplicated here with different arguments for easy access via the alias (allowing multiple calls to the same function from a single config file call).
​
For plots that load a Tensorflow or Keras model, after that model is used you will need to call `K.clear_session()` to ensure that there is no lingering session or graph. This is called after every plot function, but when loading multiple Tensorflow models this will need to be called inside the plotting function.
​
All plotting functions have a save argument to allow plots to be shown on the screen or saved, though this defaults to `True`. For uniform parameters, when saving use the `save_fig()` function that calls the usual `fig.savefig` function in matplotlib. When loading models, do this through the `utils.load_model()` function. For defining the saving and loading for a _CustomModel_, see the section below about adding models.
​
If the model has a useful hook to SHAP e.g. via the _TreeExplainer_, then make sure it is added in `utils.select_explainer()`.
​
There is a `plotting.pretty_names()` function that specifies some better looking names for the aliases of the scorers and models. Either for new models/scorers or existing ones, change the values there to affect the text on plots.
​

## Adding a new model

To add a new model, the parameter definitions need to added to `model_params.py`, which has separate dictionaries for parameter definitions for grid or random search, as well as a single model. Similarly, new models need to be added to `models.define_models()`, which includes any specific modifications for the parameter definitions for classification or regression. If larger changes are needed, you should add a separate model specifically for the problem type.
​

### CustomModel

In addition to the above, if the model is not part of scikit-learn, then it can be added as a subclass of the _CustomModel_ class (in `custom_model.py`). The methods of the base class show what needs to be defined in order for it to behave similarly to a sklearn model.
​
The key things to keep in mind are a way to save and load models, which may require temporarily deleting attributes that cannot be pickled e.g. a Tensorflow graph. Thus, when loading, these attributes will need to be added back in, e.g. by defining the graph again. If you encounter errors, first look at how the other subclasses (`MLPEnsemble` wrapping Tensorflow, and `MLPKeras` wrapping Keras) handled it.
​
Each subclass should has a `nickname` class attribute, which is the model's alias used in the config files. This is automatically taken and stored in `CustomModel.custom_aliases`, which is then used throughout when these models need to be handled differently from the normal sklearn models.
​

## Adding a new scorer

To add a new measure, simply register the function in the dictionary in `models.define_scorers()`.
​
The only caveat here is that the sklearn convention is that a higher value is better. This convention is used in the hyperparameter tuning, and so when specifying a loss or an error, then when calling `make_scorer()` then you need to pass `greater_is_better=False`. In this case, the values become negative, so when plotting the absolute value needs to be taken (this can also be done for the .csv results if desired, but is not currently).
