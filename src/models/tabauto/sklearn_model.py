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

from .base_model import BaseModel
import autosklearn.classification
import autosklearn.regression
from sklearn.multioutput import MultiOutputRegressor
import joblib
from utils.vars import CLASSIFICATION, REGRESSION


def to_matrix(data, n):
    return [data[i : i + n] for i in range(0, len(data), n)]


class SKLearnModel(BaseModel):
    def __init__(
        self,
        input_dim,
        output_dim,
        dataset_type=REGRESSION,
        method="Auto",
        multi=True,
        config=None,
        random_state=123,
    ):
        super().__init__(input_dim, output_dim, dataset_type)

        method = method.lower()
        self.method = method
        self.config = config if config else {}
        self.random_state = random_state

        if dataset_type == CLASSIFICATION:
            if method == "Auto".lower():
                print("AutoSKLearn: self.config=", self.config)
                estimators_to_use = self.config.get("estimators", ["random_forest"])
                time_left_for_this_task = self.config.get(
                    "time_left_for_this_task", 120
                )
                per_run_time_limit = self.config.get("per_run_time_limit", 30)
                memory_limit = self.config.get("memory_limit", 65536)
                n_jobs = self.config.get("n_jobs", 1)
                ensemble_size = self.config.get("ensemble_size", 1)
                cv_folds = self.config.get("cv_folds", 0)

                if cv_folds > 1:
                    kwargs = {"ensemble_size": ensemble_size, "cv_folds": cv_folds}
                else:
                    kwargs = {}

                print("e = ", estimators_to_use)
                print("time_left_for_this_task=", time_left_for_this_task)
                print("n_jobs=", n_jobs)
                print("kwargs=", kwargs)

                preprocessing_to_use = ["no_preprocessing"]
                base_model = autosklearn.classification.AutoSklearnClassifier(
                    time_left_for_this_task=time_left_for_this_task,
                    per_run_time_limit=per_run_time_limit,
                    memory_limit=memory_limit,
                    n_jobs=n_jobs,
                    include={
                        "classifier": estimators_to_use,
                        "feature_preprocessor": preprocessing_to_use,
                    },
                    delete_tmp_folder_after_terminate=True,
                    ensemble_size=ensemble_size,
                    smac_scenario_args={
                        "deterministic": "true",
                    },
                    initial_configurations_via_metalearning=0,
                    **kwargs
                )
            else:
                raise Exception("Unknown sklearn classification method")

            model = base_model

        elif dataset_type == REGRESSION:
            if method == "Auto".lower():
                print("AutoSKLearn: self.config=", self.config)
                estimators_to_use = self.config.get("estimators", ["random_forest"])
                time_left_for_this_task = self.config.get(
                    "time_left_for_this_task", 120
                )
                per_run_time_limit = self.config.get("per_run_time_limit", 30)
                memory_limit = self.config.get("memory_limit", 65536)
                n_jobs = self.config.get("n_jobs", 1)
                ensemble_size = self.config.get("ensemble_size", 1)
                cv_folds = self.config.get("cv_folds", 0)

                if cv_folds > 1:
                    kwargs = {"ensemble_size": ensemble_size, "cv_folds": cv_folds}
                else:
                    kwargs = {}

                print("e = ", estimators_to_use)
                print("time_left_for_this_task=", time_left_for_this_task)
                print("n_jobs=", n_jobs)
                print("kwargs=", kwargs)

                preprocessing_to_use = ["no_preprocessing"]
                base_model = autosklearn.regression.AutoSklearnRegressor(
                    time_left_for_this_task=time_left_for_this_task,
                    per_run_time_limit=per_run_time_limit,
                    memory_limit=memory_limit,
                    n_jobs=n_jobs,
                    include={
                        "regressor": estimators_to_use,
                        "feature_preprocessor": preprocessing_to_use,
                    },
                    delete_tmp_folder_after_terminate=True,
                    ensemble_size=ensemble_size,
                    smac_scenario_args={
                        "deterministic": "true",
                    },
                    initial_configurations_via_metalearning=0,
                    **kwargs
                )
            else:
                raise Exception("Unknown sklearn regression method")

            if output_dim > 1 and multi is True:
                model = MultiOutputRegressor(base_model)
            else:
                model = base_model
        else:
            raise Exception("Unknown dataset/problem type")

        self.model = model

    def fit_data(self, trainX, trainY, testX=None, testY=None, input_list=None):
        print("training AutoSKL model...")
        self.model.fit(trainX, trainY)

        if self.method == "Auto".lower():
            print("autosklearn/stats:", self.model.sprint_statistics())

    def predict(self, x):
        # make predictions on the testing data
        print("sklearn: predicting values ...")

        y_pred = self.model.predict(x)

        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred

    def save(self, path):
        if path:
            joblib.dump(self, "{}".format(path))
