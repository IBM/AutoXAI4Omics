# Copyright 2024 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import pandas as pd
import tensorflow
import joblib
from sklearn.preprocessing import OneHotEncoder
import logging
from utils.vars import CLASSIFICATION, REGRESSION

from models.tabauto.keras_model import KerasModel
from models.tabauto.lgbm_model import LGBMModel
from models.tabauto.xgboost_model import XGBoostModel

omicLogger = logging.getLogger("OmicLogger")


class CustomModel:
    """
    Custom model class that we use for our models.
    """

    # Aliases for model names for this class
    # Easier to maintain this manually for now
    custom_aliases = {}
    # Having a reference to the experiment folder is useful
    experiment_folder = None
    config_dict = None
    verbose = False
    model = None

    def __init__(
        self,
        random_state=None,
        scorer_func=None,
        data=None,
        data_test=None,
        labels=None,
        labels_test=None,
        n_classes=None,
        n_examples=None,
        n_dims=None,
        onehot_encode_obj=None,
        classes_=None,
        model=None,
    ):
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
        if self.config_dict["problem_type"] == CLASSIFICATION:
            # One-hot encoding has already been done, so take the info from there
            self.n_classes = self.labels.shape[1]
        elif self.config_dict["problem_type"] == REGRESSION:
            self.n_classes = 1
        # Define the model
        self._define_model()

        x_train = self.data
        y_train = self.labels
        x_val = self.data_test
        y_val = self.labels_test

        self.model.fit_data(x_train, y_train, x_val, y_val)

        if self.verbose:
            omicLogger.debug("Model Summary:")
            omicLogger.debug(self.model.summary())
        return self

    def predict(self, data):
        if self.config_dict["problem_type"] == CLASSIFICATION:
            pred_inds = np.argmax(self.model.predict_proba(data), axis=1)
            preds = self.onehot_encode_obj.categories_[0][pred_inds]
        elif self.config_dict["problem_type"] == REGRESSION:
            preds = self.model.predict(data)
        else:
            raise NotImplementedError()
        return preds.flatten()

    def predict_proba(self, data):
        if self.config_dict["problem_type"] == CLASSIFICATION:
            return self.model.predict_proba(data)
        else:
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
                omicLogger.debug(f"{key} is not a valid attribute of {self}")
        return self

    def get_params(self, deep=False):
        """
        Getter function. Vastly inferior to sklearn's, but we don't really use it.

        If issues occur, then inheriting from the BaseEstimator
        """
        return self.__dict__

    def save_model(self):
        path = self.experiment_folder / "models" / f"{self.nickname}_best"
        fname = f"{path}"
        omicLogger.debug("custom save_model: {}".format(fname))
        self.model.save(fname + ".h5")
        self._pickle_member(fname)

    @classmethod
    def load_model(cls, model_path):
        model_path = str(model_path)
        # Load the pickled instance
        with open(model_path + ".pkl", "rb") as f:
            model = joblib.load(f)
        # Load the model and set this to the relevant attribute
        omicLogger.debug("loading: {}.h5".format(model_path))
        model.model = joblib.load(model_path + ".h5")
        return model

    @classmethod
    def setup_custom_model(
        cls,
        config_dict,
        experiment_folder,
        model_name,
        param_ranges,
        scorer_func,
        x_test=None,
        y_test=None,
    ):
        """ "
        Some initial preprocessing is needed for the class to set some attributes and determine tuning type.
        """
        cls.setup_cls_vars(config_dict, experiment_folder)
        # Add the test data if provided
        if x_test is not None:
            param_ranges["data_test"] = x_test
        if y_test is not None:
            param_ranges["labels_test"] = y_test

        param_ranges["scorer_func"] = scorer_func
        return param_ranges

    @classmethod
    def setup_cls_vars(cls, config_dict, experiment_folder):
        # Refer to the config_dict in the class
        cls.config_dict = config_dict
        # Give access to the experiment_folder
        cls.experiment_folder = experiment_folder

    def __repr__(self):
        return (
            f"{self.__class__.__name__} model with params:"
            + f"{ {k:v for k,v in self.__dict__.items() if 'data' not in k if 'label' not in k} }"
        )

    def _define_model(self):
        """
        Define underlying mode/method
        """

        num_inputs = self.n_dims
        num_outputs = self.n_classes
        dataset_type = self.config_dict["problem_type"]

        config = self.config_dict.get(f"{self.nickname.lower()}_config", None)
        omicLogger.debug(f"{self.nickname.lower()}_config = {config}")

        if config:
            self.verbose = config.get("verbose", False)

        model = MODEL_REF[self.nickname](
            num_inputs,
            num_outputs,
            dataset_type=dataset_type,
            method=METHOD_REF[self.nickname],
            config=config,
            random_state=self.random_state,
        )

        # Assign the model
        self.model = model

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
        with open(fname + ".pkl", "wb") as f:
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
            self.onehot_encode_obj = OneHotEncoder(categories="auto", sparse=False)
            # Fit transform the labels that we have
            # Reshape the labels just in case (if they are, it has no effect)
            self.labels = self.onehot_encode_obj.fit_transform(
                self.labels.reshape(-1, 1)
            )
            # Set the classes for the model (useful for the plotting e.g. confusion matrix)
            self.classes_ = self.onehot_encode_obj.categories_[0]
            # Transform the test labels if we have them
            if self.labels_test is not None:
                self.labels_test = self.onehot_encode_obj.transform(
                    self.labels_test.reshape(-1, 1)
                )

    def _preparation(self):
        # Convert the labels if a DataFrame/Series
        if isinstance(self.labels, (pd.DataFrame, pd.Series)):
            self.labels = self.labels.values
        # Same for the test labels
        if self.labels_test is not None:
            if isinstance(self.labels_test, (pd.DataFrame, pd.Series)):
                self.labels_test = self.labels_test.values
        # Check if we need to one-hot encode
        if self.config_dict["problem_type"] == CLASSIFICATION:
            self._onehot_encode()
        elif self.config_dict["problem_type"] == REGRESSION:
            self.labels = self.labels.reshape(-1, 1)
            if self.labels_test is not None:
                self.labels_test = self.labels_test.reshape(-1, 1)


class FixedKeras(CustomModel):
    nickname = "FixedKeras"
    # Attributes from the config

    def __init__(
        self,
        random_state=None,
        scorer_func=None,
        data=None,
        data_test=None,
        labels=None,
        labels_test=None,
        n_classes=None,
        n_examples=None,
        n_dims=None,
        onehot_encode_obj=None,
        classes_=None,
        model=None,
        n_epochs=None,
        batch_size=None,
        lr=None,
        layer_dict=None,
        verbose=None,
        n_blocks=None,
        dropout=None,
    ):
        super().__init__(
            random_state,
            scorer_func,
            data,
            data_test,
            labels,
            labels_test,
            n_classes,
            n_examples,
            n_dims,
            onehot_encode_obj,
            classes_,
            model,
        )

        # Param attributes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr  # Learning rate
        self.layer_dict = layer_dict
        self.verbose = verbose
        self.n_blocks = n_blocks
        self.dropout = dropout

    def predict(self, data):
        if self.config_dict["problem_type"] == CLASSIFICATION:
            yp = self.model.predict(data)
            pred_inds = np.argmax(yp, axis=1)
            preds = self.onehot_encode_obj.categories_[0][pred_inds]
        elif self.config_dict["problem_type"] == REGRESSION:
            preds = self.model.predict(data)
        else:
            raise NotImplementedError()
        return preds.flatten()

    def predict_proba(self, data):
        if self.config_dict["problem_type"] == CLASSIFICATION:
            return self.model.predict(data)
        else:
            raise NotImplementedError()

    @classmethod
    def load_model(cls, model_path):
        model_path = str(model_path)
        # Load the pickled instance
        with open(model_path + ".pkl", "rb") as f:
            model = joblib.load(f)
        # Load the model with Keras and set this to the relevant attribute
        omicLogger.debug(f"loading: {model_path}.h5")
        model.model = tensorflow.keras.models.load_model(model_path + ".h5")
        return model


class AutoKeras(CustomModel):
    nickname = "AutoKeras"
    # Attributes from the config

    def predict_proba(self, data):
        if self.config_dict["problem_type"] == CLASSIFICATION:
            return self.model.predict(data)
        else:
            raise NotImplementedError()

    def predict(self, data):
        if self.config_dict["problem_type"] == CLASSIFICATION:
            pred_inds = np.argmax(self.model.predict(data), axis=1)
            preds = self.onehot_encode_obj.categories_[0][pred_inds]
        elif self.config_dict["problem_type"] == REGRESSION:
            preds = self.model.predict(data)
        else:
            raise NotImplementedError()
        return preds.flatten()

    @classmethod
    def load_model(cls, model_path):
        model_path = str(model_path)
        # Load the pickled instance
        with open(model_path + ".pkl", "rb") as f:
            model = joblib.load(f)
        # Load the model with Keras and set this to the relevant attribute
        omicLogger.debug("loading: {model_path}.h5")
        model.model = tensorflow.keras.models.load_model(model_path + ".h5")
        return model


class AutoLGBM(CustomModel):
    nickname = "AutoLGBM"
    # Attributes from the config


class AutoXGBoost(CustomModel):
    nickname = "AutoXGBoost"
    # Attributes from the config


METHOD_REF = {
    "AutoXGBoost": "train_ml_xgboost_auto",
    "AutoLGBM": "train_ml_lgbm_auto",
    "AutoKeras": "train_dnn_autokeras",
    "FixedKeras": "train_dnn_keras",
}

MODEL_REF = {
    "AutoXGBoost": XGBoostModel,
    "AutoLGBM": LGBMModel,
    "AutoKeras": KerasModel,
    "FixedKeras": KerasModel,
}
