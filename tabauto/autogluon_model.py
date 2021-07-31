import os
import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from .base_model import BaseModel

# To avoid the conflict with autogluon, which uses newer versions of some common packages,
# we can disable the package verification procedure in the following files:
# a) python3.6/site-packages/autosklearn/__init__.py
# b) python3.6/site-packages/smac/__init__.py


def to_matrix(data, n):
    return [data[i:i+n] for i in range(0, len(data), n)]


class AutogluonModel(BaseModel):

    def __init__(self, input_dim, output_dim, dataset_type, method='train_ml_autogluon', config=None):

        self.method = method
        self.savedir = None
        self.config = config if config else {}
        super().__init__(input_dim, output_dim, dataset_type)
        if dataset_type == "regression":
            if output_dim > 1:
                raise NotImplementedError

        self.model = None

    def fit_data(self, trainX, trainY, testX=None, testY=None, input_list=None):
        print("training Autogluon model...")
        if self.dataset_type == "classification":
            trainY = np.argmax(trainY, axis=-1)
            if testY is not None:
                testY = np.argmax(testY, axis=-1)

        df_x = pd.DataFrame(data=trainX)
        df_y = pd.DataFrame(data=trainY)
        df = pd.concat([df_x, df_y], axis=1, ignore_index=True)
        label_column = len(df.columns)-1

        train_data = TabularDataset(data=df)
        savedir = 'ag_models_{}/'.format(os.getpid())  # where to save trained models
        self.savedir = savedir

        auto_stack = self.config.get("auto_stack", False)
        time_limits = self.config.get("time_limits", 120)
        if self.dataset_type == "classification":

            self.model = TabularPredictor(label=label_column, 
                                  path=savedir,
                                  problem_type='multiclass'
                                  ).fit(train_data=train_data,
                                    excluded_model_types=['NN', 'CAT', 'FASTAI'],
                                    auto_stack=auto_stack, 
                                    time_limit=time_limits,
                                    keep_only_best=True)



        else:
            # https://auto.gluon.ai/api/autogluon.task.html, autogluon.tabular.TabularPrediction.fit
            # available_metrics = ['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error',
            # 'median_absolute_error', 'r2']

            self.model = TabularPredictor(label=label_column, 
                                  path=savedir,
                                  problem_type='regression', 
                                  eval_metric='mean_absolute_error'
                                  ).fit(train_data=train_data,
                                    excluded_model_types=['NN', 'CAT', 'FASTAI'],
                                    auto_stack=auto_stack, 
                                    time_limit=time_limits,
                                    keep_only_best=True)

            # nthreads_per_trial=1
            # not used: hyperparameter_tune=False, num_trials=100, search_strategy = search_strategy
        _ = self.model.fit_summary()

    def predict(self, x):
        # make predictions on the testing data
        print("autogluon: predicting values ...")
        df_x = pd.DataFrame(data=x)
        test_data = TabularDataset(data=df_x)

        y_pred = self.model.predict(test_data, as_pandas=False)
        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred

    def save(self, path):
        if path:
            import shutil
            shutil.rmtree(path, ignore_errors=True)
            shutil.copytree(self.savedir, path)
            # os.rename(self.savedir, path)
