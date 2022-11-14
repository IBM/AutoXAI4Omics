# --------------------------------------------------------------------------
# Licensed Materials - Property of IBM
#
# (C) Copyright IBM Corp. 2019, 2020
# --------------------------------------------------------------------------

import numpy as np
import pandas as pd
import tensorflow
import autokeras
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sys

class CustomModel:
    """
    Custom model class that we use for our models. This should probably inherit from sklearn's base estimator, but it doesn't.

    It doesn't because we have more control this way, and aren't subject to their verbosity/complexity.

    This class contains the poor person's abstract methods - instead of using the ABCMeta and decorator, we're raising errors.

    All of these methods are required. All of the attributes put here are required. All future subclasses need these.
    """
    # Easy reference to the config dict
    config_dict = None
    # Aliases for model names for this class
    # Easier to maintain this manually for now
    custom_aliases = {}
    # Having a reference to the experiment folder is useful
    experiment_folder = None

    def __init__(self, **kwargs):
        # References to test data
        # This is useful for tracking NN performance during training
        # and the sklearn .fit does not take two sets of data, so you cannot track test while training
        # this circumvents that if provided
        self.data_test = None
        self.labels_test = None
        self.scorer_func = None  # For tracking performance
        # Required attribute for confusion matrix and other things in sklearn
        self.classes_ = None
        # Useful to have
        self.n_dims = None
        self.n_classes = None
        self.n_examples = None

    def fit(self, data, labels, save_best=True):
        """
        sklearn style fit function

        We add an additional argument "save_best" so that we can save a model during training, and then switch saving off for plotting
        """
        raise NotImplementedError()

    def predict(self, data):
        """
        Function to predict labels or values
        """
        raise NotImplementedError()

    def predict_proba(self, data):
        """
        Function for classification to predict class probabilities
        """
        raise NotImplementedError()

    def set_params(self, **params):
        """
        Function used for setting parameters in both a tuning and single model setting.

        Universal so can implement here.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"{key} is not a valid attribute of {self}")
        return self

    def get_params(self, deep=False):
        """
        Getter function. Vastly inferior to sklearn's, but we don't really use it.

        If issues occur, then inheriting from the BaseEstimator
        """
        return self.__dict__

    def save_model(self):
        """
        Save the model. This may require some additional functions to deal with pickle (especially for TF).
        """
        raise NotImplementedError()

    @classmethod
    def load_model(cls, model_path):
        """
        Load the pickle'd model from the given path.
        """
        raise NotImplementedError()

    @classmethod
    def setup_custom_model(cls, config_dict, experiment_folder, model_name, ref_model_dict, param_ranges, scorer_func, x_test=None, y_test=None):
        """"
        Some initial preprocessing is needed for the class to set some attributes and determine tuning type.
        """
        cls.setup_cls_vars(config_dict, experiment_folder)
        # Add the test data if provided
        if x_test is not None:
            param_ranges["data_test"] = x_test
        if y_test is not None:
            param_ranges["labels_test"] = y_test

        param_ranges["scorer_func"] = scorer_func
        # Determine whether we can use hyper_tuning or not
        try:
            ref_model_dict[model_name]
            single_model_flag = False
        except KeyError:
            single_model_flag = True
            print(f"No parameter definition for {model_name} using {config_dict['hyper_tuning']}, using single model instead")
        return single_model_flag, param_ranges

    @classmethod
    def setup_cls_vars(cls, config_dict, experiment_folder):
        # Refer to the config_dict in the class
        cls.config_dict = config_dict
        # Give access to the experiment_folder
        cls.experiment_folder = experiment_folder

    def __repr__(self):
        return f"{self.__class__.__name__} model with params:{ {k:v for k,v in self.__dict__.items() if 'data' not in k if 'label' not in k} }"


class TabAuto(CustomModel):
    nickname = "tab_auto"
    # Attributes from the config
    config_dict = None
    verbose = False
    model = None

    def __init__(self, **kwargs):
        raise NotImplementedError()

    def fit(self, data, labels, save_best=True):
        raise NotImplementedError()

    def save_model(self):
        raise NotImplementedError()

    @classmethod
    def load_model(cls, model_path):
        raise NotImplementedError()

    def _define_model(self):
        """
        Define underlying mode/method
        """
        raise NotImplementedError()

    def predict(self, data):
        if self.config_dict["problem_type"] == "classification":
            pred_inds = np.argmax(self.model.predict_proba(data), axis=1)
            preds = self.onehot_encode_obj.categories_[0][pred_inds]
        elif self.config_dict["problem_type"] == "regression":
            preds = self.model.predict(data)
        else:
            raise NotImplementedError()
        return preds.flatten()

    def predict_proba(self, data):
        if self.config_dict["problem_type"] == "classification":
            return self.model.predict_proba(data)
        else:
            raise NotImplementedError()

    def set_params(self, **params):
        """
        Required function (in sklearn BaseEstimator) used for setting parameters in both a tuning and single model setting
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"{key} is not a valid attribute of {self}")
        return self

    def _pickle_member(self, fname):
        """
        Custom function to pickle an instance.
        """
        # Create a temp container
        temp_params = {}
        # Loop over the TF attributes to set to None for pickle
        for attr in ["model"]:
            temp_params[attr] = getattr(self, attr)
            setattr(self, attr, None)
        # Pickle the now TF-free object
        with open(fname+".pkl", 'wb') as f:
            joblib.dump(self, f)
        # Restore attributes
        for attr, value in temp_params.items():
            setattr(self, attr, value)

    def _onehot_encode(self):
        """
        Our network requires one-hot encoded class labels, so do that if it isn't done already
        """
        # Check if the labels are already one-hot encoded
        if len(self.labels.shape) == 1 or self.labels.shape[1] > 1:
            # Create the encode object
            self.onehot_encode_obj = OneHotEncoder(categories='auto', sparse=False)
            # Fit transform the labels that we have
            # Reshape the labels just in case (if they are, it has no effect)
            self.labels = self.onehot_encode_obj.fit_transform(self.labels.reshape(-1, 1))
            # Set the classes for the model (useful for the plotting e.g. confusion matrix)
            self.classes_ = self.onehot_encode_obj.categories_[0]
            # Transform the test labels if we have them
            if self.labels_test is not None:
                self.labels_test = self.onehot_encode_obj.transform(self.labels_test.reshape(-1, 1))

    def _preparation(self):
        # Convert the labels if a DataFrame/Series
        if isinstance(self.labels, (pd.DataFrame, pd.Series)):
            self.labels = self.labels.values
        # Same for the test labels
        if self.labels_test is not None:
            if isinstance(self.labels_test, (pd.DataFrame, pd.Series)):
                self.labels_test = self.labels_test.values
        # Check if we need to one-hot encode
        if self.config_dict["problem_type"] == "classification":
            self._onehot_encode()
        elif self.config_dict["problem_type"] == "regression":
            self.labels = self.labels.reshape(-1, 1)
            if self.labels_test is not None:
                self.labels_test = self.labels_test.reshape(-1, 1)


class FixedKeras(TabAuto):
    nickname = "fixedkeras"
    # Attributes from the config
    config_dict = None

    def __init__(self, n_epochs=None, batch_size=None, lr=None, layer_dict=None,
                 verbose=None, random_state=None, n_blocks=None, dropout=None,
                 scorer_func=None, data=None, data_test=None, labels=None, labels_test=None,
                 n_classes=None, n_examples=None, n_dims=None, onehot_encode_obj=None,
                 classes_=None, model=None):
        # Param attributes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr  # Learning rate
        self.layer_dict = layer_dict
        self.verbose = verbose
        self.random_state = random_state
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.scorer_func = scorer_func
        # Attributes for the model
        self.data = data
        self.data_test = data_test  # To track performance over epochs
        self.labels = labels
        self.labels_test = labels_test
        self.n_classes = n_classes
        self.n_examples = n_examples
        self.n_dims = n_dims
        self.onehot_encode_obj = onehot_encode_obj
        self.classes_ = classes_
        # Keras attributes
        self.model = model

    def fit(self, data, labels, save_best=True):
        """
        Fit to provided data and labels
        """
        # Setup some of the attributes from the data
        self.data = data
        self.labels = labels
        self.n_examples = data.shape[0]
        self.n_dims = data.shape[1]
        # Set up the needed things for training now we have access to the data and labels
        self._preparation()
        # Determine the number of classes
        if self.config_dict["problem_type"] == "classification":
            # One-hot encoding has already been done, so take the info from there
            self.n_classes = self.labels.shape[1]
        elif self.config_dict["problem_type"] == "regression":
            self.n_classes = 1
        # Define the model
        self._define_model()

        # Set the verbosity level
        if self.verbose:
            verbose = 1
        else:
            verbose = 0

        # x_train, x_val, y_train, y_val = train_test_split(self.data, self.labels, test_size=0.20, random_state=42)
        x_train = self.data
        y_train = self.labels
        x_val = self.data_test
        y_val = self.labels_test

        self.model.fit_data(x_train, y_train, x_val, y_val)

        if verbose:
            print("Model Summary:")
            print(self.model.summary())
        return self

    def _define_model(self):
        """
        Define underlying mode/method
        """
        from tabauto.keras_model import KerasModel

        num_inputs = self.n_dims
        num_outputs = self.n_classes
        dataset_type = self.config_dict["problem_type"]

        model = KerasModel(num_inputs, num_outputs, dataset_type=dataset_type, method="train_dnn_keras", random_state=self.random_state)

        # Assign the model
        self.model = model

    def predict(self, data):
        if self.config_dict["problem_type"] == "classification":
            yp = self.model.predict(data)
            pred_inds = np.argmax(yp, axis=1)
            preds = self.onehot_encode_obj.categories_[0][pred_inds]
        elif self.config_dict["problem_type"] == "regression":
            preds = self.model.predict(data)
        else:
            raise NotImplementedError()
        return preds.flatten()
    
    def predict_proba(self, data):
        if self.config_dict["problem_type"] == "classification":
            return self.model.predict(data)
        else:
            raise NotImplementedError()
            
    def save_model(self):
        fname = f"{self.experiment_folder / 'models' / 'fixedkeras_best'}"
        print("custom save_model: {}".format(fname))
        self.model.save(fname+".h5")
        self._pickle_member(fname)

    @classmethod
    def load_model(cls, model_path):
        model_path = str(model_path)
        # Load the pickled instance
        with open(model_path+".pkl", 'rb') as f:
            model = joblib.load(f)
        # Load the model with Keras and set this to the relevant attribute
        model.model = tensorflow.keras.models.load_model(model_path+".h5")
        return model


class AutoKeras(TabAuto):
    nickname = "autokeras"
    # Attributes from the config
    config_dict = None

    def __init__(self, random_state=None,
                 scorer_func=None, data=None, data_test=None, labels=None, labels_test=None,
                 n_classes=None, n_examples=None, n_dims=None, onehot_encode_obj=None,
                 classes_=None, model=None):
        # Param attributes
        self.random_state = random_state
        self.scorer_func = scorer_func
        # Attributes for the model
        self.data = data
        self.data_test = data_test  # To track performance over epochs
        self.labels = labels
        self.labels_test = labels_test
        self.n_classes = n_classes
        self.n_examples = n_examples
        self.n_dims = n_dims
        self.onehot_encode_obj = onehot_encode_obj
        self.classes_ = classes_
        # Keras attributes
        self.model = model

    def fit(self, data, labels, save_best=True):
        """
        Fit to provided data and labels
        """
        # Setup some of the attributes from the data
        self.data = data
        self.labels = labels
        self.n_examples = data.shape[0]
        self.n_dims = data.shape[1]
        # Set up the needed things for training now we have access to the data and labels
        self._preparation()
        # Determine the number of classes
        if self.config_dict["problem_type"] == "classification":
            # One-hot encoding has already been done, so take the info from there
            self.n_classes = self.labels.shape[1]
        elif self.config_dict["problem_type"] == "regression":
            self.n_classes = 1
        # Define the model
        self._define_model()

        # Set the verbosity level
        if self.verbose:
            verbose = 1
        else:
            verbose = 0

        x_train, x_val, y_train, y_val = train_test_split(self.data, self.labels, test_size=0.20, random_state=42)
        
        self.model.fit_data(x_train, y_train, x_val, y_val)

        if verbose:
            print("Model Summary:")
            print(self.model.summary())
        return self

    def _define_model(self):
        """
        Define underlying mode/method
        """
        from tabauto.keras_model import KerasModel

        num_inputs = self.n_dims
        num_outputs = self.n_classes
        dataset_type = self.config_dict["problem_type"]

        config = self.config_dict.get("autokeras_config", None)
        print("autokeras_config=", config)
        if config: self.verbose = config.get("verbose", False)
        model = KerasModel(num_inputs, num_outputs, dataset_type=dataset_type, method="train_dnn_autokeras", config=config, random_state=self.random_state)

        # Assign the model
        self.model = model

    def predict_proba(self, data):
        if self.config_dict["problem_type"] == "classification":
            return self.model.predict(data)
        else:
            raise NotImplementedError()

    def predict(self, data):
        if self.config_dict["problem_type"] == "classification":
            pred_inds = np.argmax(self.model.predict(data), axis=1)
            preds = self.onehot_encode_obj.categories_[0][pred_inds]
        elif self.config_dict["problem_type"] == "regression":
            preds = self.model.predict(data)
        else:
            raise NotImplementedError()
        return preds.flatten()

    def save_model(self):
        fname = f"{self.experiment_folder / 'models' / 'autokeras_best'}"
        print("custom save_model: {}".format(fname))
        self.model.save(fname+".h5")
        self._pickle_member(fname)

    @classmethod
    def load_model(cls, model_path):
        model_path = str(model_path)
        # Load the pickled instance
        with open(model_path+".pkl", 'rb') as f:
            model = joblib.load(f)
        # Load the model with Keras and set this to the relevant attribute
        print("loading:", model_path+".h5")
        # custom_objects = {"cast_to_float32": autokeras.keras_layers.CastToFloat32}
        # model.model = tensorflow.keras.models.load_model(model_path+".h5", custom_objects=custom_objects)
        model.model = tensorflow.keras.models.load_model(model_path+".h5")
        return model


class AutoSKLearn(TabAuto):
    nickname = "autosklearn"
    # Attributes from the config
    config_dict = None

    def __init__(self, random_state=None, scorer_func=None, data=None, data_test=None, labels=None, labels_test=None,
                 n_classes=None, n_examples=None, n_dims=None, onehot_encode_obj=None, classes_=None, model=None):
        # Param attributes
        self.random_state = random_state
        self.scorer_func = scorer_func
        # Attributes for the model
        self.data = data
        self.data_test = data_test  # To track performance over epochs
        self.labels = labels
        self.labels_test = labels_test
        self.n_classes = n_classes
        self.n_examples = n_examples
        self.n_dims = n_dims
        self.onehot_encode_obj = onehot_encode_obj
        self.classes_ = classes_
        # Keras attributes
        self.model = model

    def fit(self, data, labels, save_best=True):
        """
        Fit to provided data and labels
        """
        # Setup some of the attributes from the data
        self.data = data
        self.labels = labels
        self.n_examples = data.shape[0]
        self.n_dims = data.shape[1]
        # Set up the needed things for training now we have access to the data and labels
        self._preparation()
        # Determine the number of classes
        if self.config_dict["problem_type"] == "classification":
            # One-hot encoding has already been done, so take the info from there
            self.n_classes = self.labels.shape[1]
        elif self.config_dict["problem_type"] == "regression":
            self.n_classes = 1
        # Define the model
        self._define_model()

        # x_train, x_val, y_train, y_val = train_test_split(self.data, self.labels, test_size=0.20, random_state=42)
        x_train = self.data
        y_train = self.labels
        x_val = self.data_test
        y_val = self.labels_test

        self.model.fit_data(x_train, y_train, x_val, y_val)

        if self.verbose:
            print("Model Summary:")
            print(self.model.summary())
        return self

    def _define_model(self):
        """
        Define underlying mode/method
        """
        from tabauto.sklearn_model import SKLearnModel

        num_inputs = self.n_dims
        num_outputs = self.n_classes
        dataset_type = self.config_dict["problem_type"]

        config = self.config_dict.get("autosklearn_config", None)
        print("autosklearn_config=", config)
        if config: self.verbose = config.get("verbose", False)
        model = SKLearnModel(num_inputs, num_outputs, dataset_type=dataset_type, method="auto", config=config, random_state=self.random_state)

        # Assign the model
        self.model = model

    def save_model(self):
        fname = f"{self.experiment_folder / 'models' / 'autosklearn_best'}"
        print("custom save_model: {}".format(fname))
        self.model.save(fname+".h5")
        self._pickle_member(fname)

    @classmethod
    def load_model(cls, model_path):
        model_path = str(model_path)
        # Load the pickled instance
        with open(model_path+".pkl", 'rb') as f:
            model = joblib.load(f)
        # Load the model and set this to the relevant attribute
        print("loading: {}.h5".format(model_path))
        model.model = joblib.load(model_path+".h5")
        return model


class AutoLGBM(TabAuto):
    nickname = "autolgbm"
    # Attributes from the config
    config_dict = None

    def __init__(self, random_state=None,
                 scorer_func=None, data=None, data_test=None, labels=None, labels_test=None,
                 n_classes=None, n_examples=None, n_dims=None, onehot_encode_obj=None,
                 classes_=None, model=None):
        # Param attributes
        self.random_state = random_state
        self.scorer_func = scorer_func
        # Attributes for the model
        self.data = data
        self.data_test = data_test  # To track performance over epochs
        self.labels = labels
        self.labels_test = labels_test
        self.n_classes = n_classes
        self.n_examples = n_examples
        self.n_dims = n_dims
        self.onehot_encode_obj = onehot_encode_obj
        self.classes_ = classes_
        #
        self.model = model

    def fit(self, data, labels, save_best=True):
        """
        Fit to provided data and labels
        """
        # Setup some of the attributes from the data
        self.data = data
        self.labels = labels
        self.n_examples = data.shape[0]
        self.n_dims = data.shape[1]
        # Set up the needed things for training now we have access to the data and labels
        self._preparation()
        # Determine the number of classes
        if self.config_dict["problem_type"] == "classification":
            # One-hot encoding has already been done, so take the info from there
            self.n_classes = self.labels.shape[1]
        elif self.config_dict["problem_type"] == "regression":
            self.n_classes = 1
        # Define the model
        self._define_model()

        # x_train, x_val, y_train, y_val = train_test_split(self.data, self.labels, test_size=0.20, random_state=42)
        x_train = self.data
        y_train = self.labels
        x_val = self.data_test
        y_val = self.labels_test

        self.model.fit_data(x_train, y_train, x_val, y_val)

        if self.verbose:
            print("Model Summary:")
            print(self.model.summary())
        return self

    def _define_model(self):
        """
        Define underlying mode/method
        """
        from tabauto.lgbm_model import LGBMModel

        num_inputs = self.n_dims
        num_outputs = self.n_classes
        dataset_type = self.config_dict["problem_type"]

        config = self.config_dict.get("autolgbm_config", None)
        print("autolgbm_config=", config)
        if config: self.verbose = config.get("verbose", False)
        model = LGBMModel(num_inputs, num_outputs, dataset_type=dataset_type, method="train_ml_lgbm_auto", config=config, random_state=self.random_state)

        # Assign the model
        self.model = model

    def save_model(self):
        fname = f"{self.experiment_folder / 'models' / 'autolgbm_best'}"
        print("custom save_model: {}".format(fname))
        self.model.save(fname+".h5")
        self._pickle_member(fname)

    @classmethod
    def load_model(cls, model_path):
        model_path = str(model_path)
        # Load the pickled instance
        with open(model_path+".pkl", 'rb') as f:
            model = joblib.load(f)
        # Load the model and set this to the relevant attribute
        print("loading: {}.h5".format(model_path))
        model.model = joblib.load(model_path+".h5")
        return model


class AutoXGBoost(TabAuto):
    nickname = "autoxgboost"
    # Attributes from the config
    config_dict = None

    def __init__(self, random_state=None, scorer_func=None,
                 data=None, data_test=None, labels=None, labels_test=None,
                 n_classes=None, n_examples=None, n_dims=None, onehot_encode_obj=None,
                 classes_=None, model=None):
        # Param attributes
        self.random_state = random_state
        self.scorer_func = scorer_func
        # Attributes for the model
        self.data = data
        self.data_test = data_test  # To track performance over epochs
        self.labels = labels
        self.labels_test = labels_test
        self.n_classes = n_classes
        self.n_examples = n_examples
        self.n_dims = n_dims
        self.onehot_encode_obj = onehot_encode_obj
        self.classes_ = classes_
        #
        self.model = model

    def fit(self, data, labels, save_best=True):
        """
        Fit to provided data and labels
        """
        # Setup some of the attributes from the data
        self.data = data
        self.labels = labels
        self.n_examples = data.shape[0]
        self.n_dims = data.shape[1]
        # Set up the needed things for training now we have access to the data and labels
        self._preparation()
        # Determine the number of classes
        if self.config_dict["problem_type"] == "classification":
            # One-hot encoding has already been done, so take the info from there
            self.n_classes = self.labels.shape[1]
        elif self.config_dict["problem_type"] == "regression":
            self.n_classes = 1
        # Define the model
        self._define_model()

        # x_train, x_val, y_train, y_val = train_test_split(self.data, self.labels, test_size=0.20, random_state=42)
        x_train = self.data
        y_train = self.labels
        x_val = self.data_test
        y_val = self.labels_test

        self.model.fit_data(x_train, y_train, x_val, y_val)

        if self.verbose:
            print("Model Summary:")
            print(self.model.summary())
        return self

    def _define_model(self):
        """
        Define underlying mode/method
        """
        from tabauto.xgboost_model import XGBoostModel

        num_inputs = self.n_dims
        num_outputs = self.n_classes
        dataset_type = self.config_dict["problem_type"]

        config = self.config_dict.get("autoxgboost_config", None)
        print("autoxgboost_config=", config)
        if config: self.verbose = config.get("verbose", False)
        model = XGBoostModel(num_inputs, num_outputs, dataset_type=dataset_type, method="train_ml_xgboost_auto", config=config, random_state=self.random_state)

        # Assign the model
        self.model = model

    def save_model(self):
        fname = f"{self.experiment_folder / 'models' / 'autoxgboost_best'}"
        print("custom save_model: {}".format(fname))
        self.model.save(fname+".h5")
        self._pickle_member(fname)

    @classmethod
    def load_model(cls, model_path):
        model_path = str(model_path)
        # Load the pickled instance
        with open(model_path+".pkl", 'rb') as f:
            model = joblib.load(f)
        # Load the model and set this to the relevant attribute
        print("loading: {}.h5".format(model_path))
        model.model = joblib.load(model_path+".h5")
        return model


# class AutoGluon(TabAuto):
#     nickname = "autogluon"
#     # Attributes from the config
#     config_dict = None

#     def __init__(self, random_state=None, scorer_func=None,
#                  data=None, data_test=None, labels=None, labels_test=None,
#                  n_classes=None, n_examples=None, n_dims=None, onehot_encode_obj=None,
#                  classes_=None, model=None):
#         # Param attributes
#         self.random_state = random_state
#         self.scorer_func = scorer_func
#         # Attributes for the model
#         self.data = data
#         self.data_test = data_test  # To track performance over epochs
#         self.labels = labels
#         self.labels_test = labels_test
#         self.n_classes = n_classes
#         self.n_examples = n_examples
#         self.n_dims = n_dims
#         self.onehot_encode_obj = onehot_encode_obj
#         self.classes_ = classes_
#         #
#         self.model = model

#     def fit(self, data, labels, save_best=True):
#         """
#         Fit to provided data and labels
#         """
#         # Setup some of the attributes from the data
#         self.data = data
#         self.labels = labels
#         self.n_examples = data.shape[0]
#         self.n_dims = data.shape[1]
#         # Set up the needed things for training now we have access to the data and labels
#         self._preparation()
#         # Determine the number of classes
#         if self.config_dict["problem_type"] == "classification":
#             # One-hot encoding has already been done, so take the info from there
#             self.n_classes = self.labels.shape[1]
#         elif self.config_dict["problem_type"] == "regression":
#             self.n_classes = 1
#         # Define the model
#         self._define_model()

#         # x_train, x_val, y_train, y_val = train_test_split(self.data, self.labels, test_size=0.20, random_state=42)
#         x_train = self.data
#         y_train = self.labels
#         x_val = self.data_test
#         y_val = self.labels_test

#         self.model.fit_data(x_train, y_train, x_val, y_val)

#         if self.verbose:
#             print("Model Summary:")
#             print(self.model.summary())
#         return self

#     def _define_model(self):
#         """
#         Define underlying mode/method
#         """
#         from tabauto.autogluon_model import AutogluonModel

#         num_inputs = self.n_dims
#         num_outputs = self.n_classes
#         dataset_type = self.config_dict["problem_type"]

#         config = self.config_dict.get("autogluon_config", None)
#         print("autogluon_config=", config)
#         if config: self.verbose = config.get("verbose", False)
#         model = AutogluonModel(num_inputs, num_outputs, dataset_type=dataset_type, method="auto", config=config)

#         # Assign the model
#         self.model = model

#     def save_model(self):
#         fname = f"{self.experiment_folder / 'models' / 'autogluon_best'}"
#         print("custom save_model: {}".format(fname))
#         self.model.save(fname+"_h5")
#         self._pickle_member(fname)

#     @classmethod
#     def load_model(cls, model_path):
#         from autogluon.tabular import TabularPredictor

#         model_path = str(model_path)
#         # Load the pickled instance
#         with open(model_path+".pkl", 'rb') as f:
#             model = joblib.load(f)
#         # Load the model and set this to the relevant attribute
#         print("loading: {}_h5".format(model_path))
#         model.model = TabularPredictor.load(model_path+"_h5")
#         return model

#     def predict_proba(self, data):
#         from autogluon.tabular import TabularDataset

#         df_x = pd.DataFrame(data=data)
#         test_data = TabularDataset(data=df_x)

#         if self.config_dict["problem_type"] == "classification":
#             preds = self.model.predict_proba(test_data)
#             if isinstance(preds, (pd.DataFrame, pd.Series)):
#                 preds = preds.values
#             return preds
#         else:
#             raise NotImplementedError()

#     def predict(self, data):
#         """
#         Function to predict labels or values
#         """
#         from autogluon.tabular import TabularDataset

#         df_x = pd.DataFrame(data=data)
#         test_data = TabularDataset(data=df_x)

#         if self.config_dict["problem_type"] == "classification":
#             # pred_inds = np.argmax(self.model.predict_proba(test_data), axis=1)
#             preds = self.model.predict_proba(test_data)
#             if isinstance(preds, (pd.DataFrame, pd.Series)):
#                 preds = preds.values
#             pred_inds = np.argmax(preds, axis=1)
#             preds = self.onehot_encode_obj.categories_[0][pred_inds]
#         elif self.config_dict["problem_type"] == "regression":
#             preds = self.model.predict(test_data)
#             if isinstance(preds, (pd.DataFrame, pd.Series)):
#                 preds = preds.values
#         else:
#             raise NotImplementedError()
#         return preds.flatten()
