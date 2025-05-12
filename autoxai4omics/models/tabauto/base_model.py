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

from abc import ABCMeta, abstractmethod
from utils.vars import CLASSIFICATION, REGRESSION
import numpy as np


class BaseModel:
    __metaclass__ = ABCMeta

    def __init__(self, input_dim, output_dim, dataset_type=REGRESSION):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = None
        self.dataset_type = dataset_type
        self.model_ohe = None

    @abstractmethod
    def fit_data(self, trainX, trainY, testX=None, testY=None, input_list=None):
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        raise NotImplementedError

    def predict(self, x):
        # make predictions on the testing data
        print("BaseModel: predicting values ...")
        y_pred = self.model.predict(x)

        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred

    def predict_proba(self, x):
        print("BaseModel: predicting probs ...")
        y_pred = self.model.predict_proba(x)
        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred

    def transform_output(self, y_pred):
        if self.model_ohe:
            y_pred = self.model_ohe.transform(y_pred)

        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        if self.dataset_type == CLASSIFICATION:
            y_pred = np.argmax(y_pred, axis=-1)

        return y_pred

    def best_params(self):
        try:
            return self.model.best_params_
        except Exception:
            return None

    def feature_importances(self):
        try:
            return self.model.feature_importances_
        except Exception:
            return None

    def summary(self):
        pass
