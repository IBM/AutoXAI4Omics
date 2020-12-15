import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputRegressor
import joblib
import lightgbm as lgb_core
import optuna
from .base_model import BaseModel


def to_matrix(data, n):
    return [data[i:i+n] for i in range(0, len(data), n)]


class LGBMObjective(object):
    def __init__(self, dataset_type, train_x, train_y, test_x, test_y):
        # Hold this implementation specific arguments as the fields of the class.
        self.dataset_type = dataset_type
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        train_x = self.train_x
        train_y = self.train_y
        valid_x = self.test_x
        valid_y = self.test_y

        param = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.1, 1),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        if self.dataset_type == 'classification':
            """
            bst = lgb_core.LGBMClassifier(**param)
            print("train_y=", train_y)
            bst.fit(train_x, train_y)
            print("valid_x=", valid_x)
            preds = bst.predict(valid_x)
            print("preds=", preds)
            pred_labels = np.rint(preds)
            score = accuracy_score(valid_y, pred_labels)
            """

            from sklearn.metrics import mean_absolute_error
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=5, shuffle=True, random_state=55)
            scores = []
            for train_index, test_index in kf.split(train_x):

                lgb_model = lgb_core.LGBMClassifier(**param)
                lgb_model.fit(train_x[train_index], train_y[train_index])
                predictions = lgb_model.predict(train_x[test_index])
                predictions = np.rint(predictions)
                actuals = train_y[test_index]
                s = accuracy_score(actuals, predictions)
                # print(s)
                scores.append(s)

        else:
            """
            param["objective"] = "regression"
            param["metric"] = "l1"
            bst = lgb_core.LGBMRegressor(**param)
            bst.fit(train_x, train_y)
            preds = bst.predict(valid_x)
            score = sklearn.metrics.mean_absolute_error(valid_y, preds)
            """

            from sklearn.metrics import mean_absolute_error
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=5, shuffle=True, random_state=55)
            scores = []
            for train_index, test_index in kf.split(train_x):
                param["objective"] = "regression"
                param["metric"] = "l1"

                lgb_model = lgb_core.LGBMRegressor(**param)
                # lgb_model.set_params(**param)
                lgb_model.fit(train_x[train_index], train_y[train_index])
                predictions = lgb_model.predict(train_x[test_index])
                actuals = train_y[test_index]
                s = mean_absolute_error(actuals, predictions)
                print(s)
                scores.append(s)

        print("Autolgbm (trial={}): cv scores = {}".format(trial.number, scores))
        min_score = np.min(scores)
        max_score = np.max(scores)
        avg_score = np.average(scores)
        std_score = np.std(scores)
        print("Autolgbm (trial={}): cv stats: min:{} max:{} avg:{} std:{}".format(trial.number, min_score, max_score, avg_score, std_score))
        cv_stats = dict(scores=scores, min_score=min_score, max_score=max_score, avg_score=avg_score, std_score=std_score)
        trial.set_user_attr("cv_stats", cv_stats)
        score = avg_score

        return score


class LGBMModel(BaseModel):

    def __init__(self, input_dim, output_dim, dataset_type, method='train_ml_lgbm', config=None):

        super().__init__(input_dim, output_dim, dataset_type)
        self.method = method
        self.config = config if config else {}
        if self.method == 'train_ml_lgbm':
            self.__init_fx__(input_dim, output_dim, dataset_type)
        elif self.method == 'train_ml_lgbm_auto':
            self.__init_optuna__(input_dim, output_dim, dataset_type)

    def __init_fx__(self, input_dim, output_dim, dataset_type):
        if dataset_type == "classification":
            base_model = lgb_core.LGBMClassifier()
            model = base_model
        else:
            base_model = lgb_core.LGBMRegressor()
            if output_dim > 1:
                model = MultiOutputRegressor(base_model)
            else:
                model = base_model

        self.model = model

    def __init_optuna__(self, input_dim, output_dim, dataset_type):
        if dataset_type == "classification":
            model = None
        else:
            if output_dim > 1:
                raise NotImplementedError
            else:
                model = None

        print("AutoLGBM: self.config=", self.config)
        self.n_trials = self.config.get('n_trials', 100)
        self.timeout = self.config.get('timeout', None)
        print("AutoLGBM: n_trials={} timeout={}".format(self.n_trials, self.timeout))

        self.model = model

    def fit_data_fx(self, trainX, trainY, testX=None, testY=None):
        print("training LGBM model...")
        if self.dataset_type == "classification":
            trainY = np.argmax(trainY, axis=-1)
            if testY is not None:
                testY = np.argmax(testY, axis=-1)

        # self.model.fit(trainX, trainY, eval_set=[(testX, testY)], eval_metric='l1', early_stopping_rounds=5)
        self.model.fit(trainX, trainY, eval_metric='l1')

    def fit_data_optuna(self, trainX, trainY, testX=None, testY=None):
        print("training LGBM model...")

        if self.dataset_type == "classification":
            trainY = np.argmax(trainY, axis=-1)
            if testY is not None:
                testY = np.argmax(testY, axis=-1)

        if self.dataset_type == 'classification':
            direction = "maximize"
        else:
            direction = "minimize"

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction=direction)
        objective = LGBMObjective(self.dataset_type, trainX, trainY, testX, testY)

        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        print('best_trial=', study.best_trial)
        print('best_params=', study.best_params)
        cv_stats = study.best_trial.user_attrs["cv_stats"]

        if self.dataset_type == 'classification':
            self.model = lgb_core.LGBMClassifier(**study.best_params)
        else:
            param = study.best_params
            param["objective"] = "regression"
            param["metric"] = "l1"
            self.model = lgb_core.LGBMRegressor(**param)
        self.model.fit(trainX, trainY)
        print("="*50)
        print("Autolgb: best model=", self.model)
        for key in cv_stats.keys():
            print("Autolgb: cv_stats[\"{}\"]={}".format(key, cv_stats[key]))
        print("="*50)


    def fit_data(self, trainX, trainY, testX=None, testY=None, input_list=None):
        if self.method == 'train_ml_lgbm':
            self.fit_data_fx(trainX, trainY, testX, testY)
        elif self.method == 'train_ml_lgbm_auto':
            self.fit_data_optuna(trainX, trainY, testX, testY)

    """
    def predict(self, x):
        # make predictions on the testing data
        print("lgbm: predict...")
        if self.dataset_type == "classification":
            y_pred = self.model.predict_proba(x)
        else:
            y_pred = self.model.predict(x)

        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred
    """

    def save(self, path):
        if path:
            # joblib.dump(self.model, '{}'.format(path))
            joblib.dump(self, '{}'.format(path))
