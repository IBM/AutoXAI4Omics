import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputRegressor
import joblib
import xgboost as xgb
import optuna
from .base_model import BaseModel


def to_matrix(data, n):
    return [data[i : i + n] for i in range(0, len(data), n)]


# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
class XGBoostObjective(object):
    def __init__(self, dataset_type, train_x, train_y, test_x=None, test_y=None, num_class=0, random_state=123):
        # Hold this implementation specific arguments as the fields of the class.
        self.dataset_type = dataset_type
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.num_class = num_class
        self.random_state = random_state

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        train_x = self.train_x
        train_y = self.train_y
        valid_x = self.test_x
        valid_y = self.test_y

        param = {
            "verbosity": 1,
            # "silent": 1,
            # "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear"]),  # , "dart"]),
            "booster": trial.suggest_categorical("booster", ["gbtree"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 0, 100),
        }

        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        if self.dataset_type == "classification":
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=5, shuffle=True, random_state=55)
            scores = []
            for train_index, test_index in kf.split(train_x):
                xgb_model = xgb.XGBClassifier(
                    objective="multi:softmax", num_class=self.num_class, n_jobs=-1, random_state=self.random_state
                )
                xgb_model.set_params(**param)
                xgb_model.fit(train_x[train_index], train_y[train_index])
                predictions = xgb_model.predict(train_x[test_index])
                predictions = np.rint(predictions)
                actuals = train_y[test_index]
                s = accuracy_score(actuals, predictions)
                print(s)
                scores.append(s)
            score = sum(scores) / len(scores)

        else:  # 'regression'
            from sklearn.metrics import mean_absolute_error
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=5, shuffle=True, random_state=55)
            scores = []
            for train_index, test_index in kf.split(train_x):
                xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=-1, random_state=self.random_state)
                xgb_model.set_params(**param)
                xgb_model.fit(train_x[train_index], train_y[train_index])
                predictions = xgb_model.predict(train_x[test_index])
                actuals = train_y[test_index]
                s = mean_absolute_error(actuals, predictions)
                print(s)
                scores.append(s)
            score = sum(scores) / len(scores)

            # xgb.cv does not work for regression

        return score


class XGBoostModel(BaseModel):
    def __init__(self, input_dim, output_dim, dataset_type, method="train_ml_xgboost", config=None, random_state=123):
        super().__init__(input_dim, output_dim, dataset_type)
        self.method = method
        self.config = config if config else {}
        self.random_state = random_state
        if self.method == "train_ml_xgboost":
            self.__init_fx__(input_dim, output_dim, dataset_type)
        elif self.method == "train_ml_xgboost_auto":
            self.__init_optuna__(input_dim, output_dim, dataset_type)

    def __init_fx__(self, input_dim, output_dim, dataset_type):
        if dataset_type == "classification":
            base_model = xgb.XGBClassifier(
                max_depth=10,
                learning_rate=0.1,
                n_estimators=100,
                verbosity=0,
                silent=True,
                objective="multi:softmax",
                # objective="binary:logistic",
                booster="gbtree",
                n_jobs=-1,
                nthread=-1,
                gamma=0,
                min_child_weight=1,
                max_delta_step=0,
                subsample=0.7,
                colsample_bytree=1,
                colsample_bylevel=1,
                colsample_bynode=1,
                reg_alpha=0,
                reg_lambda=1,
                scale_pos_weight=1,
                base_score=0.5,
                random_state=self.random_state,
                # seed=0
            )
            model = base_model
        else:
            base_model = xgb.XGBRegressor(
                objective="reg:squarederror",
                learning_rate=0.01,
                max_depth=10,
                alpha=10,
                n_estimators=100,
                random_state=self.random_state,
            )
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

        print("AutoXGB: self.config=", self.config)
        self.n_trials = self.config.get("n_trials", 100)
        self.timeout = self.config.get("timeout", None)
        print("AutoXGB: n_trials={} timeout={}".format(self.n_trials, self.timeout))

        self.output_dim = output_dim
        self.model = model

    def fit_data_fx(self, trainX, trainY, testX=None, testY=None):
        print("training XGBoost model...")

        if self.dataset_type == "classification":
            n_classes = trainY.shape[1]
            trainY = np.argmax(trainY, axis=-1)
            params = self.model.get_params()
            params["num_class"] = n_classes
            self.model.set_params(**params)

        self.model.fit(trainX, trainY)

    def fit_data_optuna(self, trainX, trainY, testX, testY):
        if self.dataset_type == "classification":
            n_classes = trainY.shape[1]
            trainY = np.argmax(trainY, axis=-1)
            if testY is not None:
                testY = np.argmax(testY, axis=-1)

        if self.dataset_type == "classification":
            direction = "maximize"
        else:
            direction = "minimize"

        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=self.random_state))
        # DOES THE BELOW NEED TO HAVE RANDOM STATE SET??
        objective = XGBoostObjective(
            self.dataset_type, trainX, trainY, testX, testY, num_class=self.output_dim, random_state=self.random_state
        )

        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        print("best_trial=", study.best_trial)
        print("best_params=", study.best_params)

        if self.dataset_type == "classification":
            self.model = xgb.XGBClassifier(
                objective="multi:softmax",
                num_class=self.output_dim,
                random_state=self.random_state,
                **study.best_params
            )
        else:
            self.model = xgb.XGBRegressor(
                objective="reg:squarederror", random_state=self.random_state, **study.best_params
            )

        self.model.fit(trainX, trainY)

    def fit_data(self, trainX, trainY, testX=None, testY=None, input_list=None):
        if self.method == "train_ml_xgboost":
            self.fit_data_fx(trainX, trainY, testX, testY)
        elif self.method == "train_ml_xgboost_auto":
            self.fit_data_optuna(trainX, trainY, testX, testY)

    def save(self, path):
        if path:
            joblib.dump(self, "{}".format(path))
