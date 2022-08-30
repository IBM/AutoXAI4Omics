from .base_model import BaseModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# To avoid the conflict with autogluon, which uses newer versions of some common packages,
# we can disable the package verification procedure in the following files:
# a) python3.6/site-packages/autosklearn/__init__.py
# b) python3.6/site-packages/smac/__init__.py

import autosklearn.classification
import autosklearn.regression

# Multioutput regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib


def to_matrix(data, n):
    return [data[i:i+n] for i in range(0, len(data), n)]


class SKLearnModel(BaseModel):
    def __init__(self, input_dim, output_dim, dataset_type="regression", method='RandomForest', multi=True,
                 config=None, random_state=123):

        super().__init__(input_dim, output_dim, dataset_type)

        method = method.lower()
        self.method = method
        self.config = config if config else {}
        self.random_state = random_state
        
        
        if dataset_type == "classification":
            if method == 'KNeighbors'.lower():
                # KNN (works for multiple outputs)
                base_model = KNeighborsClassifier(3,random_state=self.random_state)
            elif method == 'DecisionTree'.lower():
                # DecisionTree (works for multiple outputs)
                max_depth = 30
                base_model = DecisionTreeClassifier(max_depth=max_depth, class_weight=None, criterion='entropy',
                                                    max_features=None, max_leaf_nodes=None,
                                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                                    min_samples_leaf=1, min_samples_split=10,
                                                    min_weight_fraction_leaf=0.0, presort=False, random_state=self.random_state,
                                                    splitter='best')
            elif method == 'RandomForest'.lower():
                # Random Forest (works for multiple outputs)
                print("RandomForestClassifier....")
                base_model = RandomForestClassifier(n_estimators=300, max_depth=25, random_state=self.random_state,
                                                    bootstrap=True, class_weight=None, criterion='gini',
                                                    max_features='auto', max_leaf_nodes=None,
                                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                                    min_samples_leaf=1, min_samples_split=10,
                                                    min_weight_fraction_leaf=0.0,  n_jobs=1,
                                                    oob_score=False, verbose=0,
                                                    warm_start=False)
            elif method == 'AdaBoost'.lower():
                base_model = AdaBoostClassifier(random_state=self.random_state)
            elif method == 'NaiveBayes'.lower():
                base_model = GaussianNB(random_state=self.random_state)
            elif method == 'Auto'.lower():
                print("AutoSKLearn: self.config=", self.config)
                estimators_to_use = self.config.get("estimators", ["random_forest"])
                time_left_for_this_task = self.config.get("time_left_for_this_task", 120)
                per_run_time_limit = self.config.get("per_run_time_limit", 30)
                memory_limit = self.config.get("memory_limit", 65536)
                n_jobs = self.config.get("n_jobs", 1)
                ensemble_size = self.config.get("ensemble_size", 1)
                cv_folds = self.config.get("cv_folds", 0)

                if cv_folds > 1:
                    kwargs = {'ensemble_size': ensemble_size,
                              'cv_folds': cv_folds}
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
                    include = {
                        'classifier': estimators_to_use,
                        'feature_preprocessor': preprocessing_to_use
                        },
                    # exclude={
                    #     'classifier':estimators_to_exclude,
                    #     'feature_preprocessor':None
                    # },
                    delete_tmp_folder_after_terminate=True,
                    # delete_output_folder_after_terminate=True,
                    ensemble_size=ensemble_size,
                    # load_models = False,
                    smac_scenario_args={
                        'deterministic': 'true',
                        },
                    initial_configurations_via_metalearning=0,
                    **kwargs)
            else:
                raise Exception("Unknown sklearn classification method")

            model = base_model

        elif dataset_type == "regression":
            if method == 'KNeighbors'.lower():
                # KNN (works for multiple outputs)
                base_model = KNeighborsRegressor(n_neighbors=2, weights='distance',random_state=self.random_state)
            elif method == 'DecisionTree'.lower():
                # DecisionTree (works for multiple outputs)
                base_model = DecisionTreeRegressor(random_state=self.random_state)
            elif method == 'RandomForest'.lower():
                # Random Forest (works for multiple outputs)
                base_model = RandomForestRegressor(n_estimators=300, criterion='mse',random_state=self.random_state)
                """
                base_model = RandomForestRegressor(bootstrap=True,
                                                    max_depth=100, max_features='auto', max_leaf_nodes=None,
                                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                                    min_samples_leaf=1, min_samples_split=10,
                                                    min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=None,
                                                    oob_score=False, random_state=None, verbose=0,
                                                    warm_start=False)
                """

            elif method == 'ExtraTrees'.lower():
                base_model = ExtraTreesRegressor(n_estimators=10, max_features=None, random_state=self.random_state)
            elif method == 'Auto'.lower():
                print("AutoSKLearn: self.config=", self.config)
                estimators_to_use = self.config.get("estimators", ["random_forest"])
                time_left_for_this_task = self.config.get("time_left_for_this_task", 120)
                per_run_time_limit = self.config.get("per_run_time_limit", 30)
                memory_limit = self.config.get("memory_limit", 65536)
                n_jobs = self.config.get("n_jobs", 1)
                ensemble_size = self.config.get("ensemble_size", 1)
                cv_folds = self.config.get("cv_folds", 0)

                if cv_folds > 1:
                    kwargs = {'ensemble_size': ensemble_size,
                              'cv_folds': cv_folds}
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
                    include = {
                        'regressor': estimators_to_use,
                        'feature_preprocessor': preprocessing_to_use
                        },
                    # exclude={
                    #     'regressor':estimators_to_exclude,
                    #     'feature_preprocessor':None
                    # },
                    # initial_configurations_via_metalearning=0,
                    delete_tmp_folder_after_terminate=True,
                    # delete_output_folder_after_terminate=True,
                    ensemble_size=ensemble_size,
                    # resampling_strategy="cv", 
                    # resampling_strategy_arguments={"folds": 5},
                    # load_models = False,
                    smac_scenario_args={
                        'deterministic': 'true',
                        },
                    initial_configurations_via_metalearning=0,
                    **kwargs)  
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

        if self.method == 'Auto'.lower():
            # print("autosklearn/cv_results=", self.model.cv_results_)
            print('autosklearn/stats:', self.model.sprint_statistics())
            # print('autosklearn/models:', self.model.show_models())
            # print("self.model.best_params_=", self.model.best_params_)

    def predict(self, x):
        # make predictions on the testing data
        print("sklearn: predicting values ...")

        y_pred = self.model.predict(x)

        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred

    def save(self, path):
        if path:
            joblib.dump(self, '{}'.format(path))
