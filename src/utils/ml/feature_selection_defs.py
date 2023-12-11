from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)


FS_KBEST_METRICS = {
    "f_regression": f_regression,
    "f_classif": f_classif,
    "mutual_info_regression": mutual_info_regression,
    "mutual_info_classif": mutual_info_classif,
}
FS_METHODS = {
    "SelectKBest": SelectKBest,
    "RFE": RFE,
}
