from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LinearRegression,
    Ridge,
    RidgeCV,
    SGDRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


MODELS = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "RidgeCV": RidgeCV,
    "SGDRegressor": SGDRegressor,
    "ElasticNet": ElasticNet,
    "ElasticNetCV": ElasticNetCV,
    "Lars": Lars,
    "LarsCV": LarsCV,
    "Lasso": Lasso,
    "LassoCV": LassoCV,
    "LassoLars": LassoLars,
    "LassoLarsCV": LassoLarsCV,
    "RandomForestRegressor": RandomForestRegressor,
    "KNeighborsRegressor": KNeighborsRegressor,
    "MLPClassifier": MLPClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
    "SVC": SVC,
    "GaussianProcessClassifier": GaussianProcessClassifier,
    "RBF": RBF,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "AdaBoostClassifier": AdaBoostClassifier,
}
