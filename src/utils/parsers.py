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

from os.path import exists
from models.model_defs import MODELS
from metrics.metric_defs import METRICS
import logging
from utils.vars import CLASSIFICATION, REGRESSION
from utils.ml.feature_selection_defs import FS_METHODS, FS_KBEST_METRICS

omicLogger = logging.getLogger("OmicLogger")


def list_type_check(givenList, typeName, entryname="list"):
    typeList = all([type(x) == typeName for x in givenList])
    if not typeList:
        raise ValueError(f"All elements in {entryname} must be of type: {typeName}")


def type_check(item, typeName, entryname="Object"):
    if type(item) != typeName:
        raise ValueError(f"{entryname} is not of type: {typeName}")


#################### plotting ####################
def parser_plotting(plotting_entry, problem):
    keys = plotting_entry.keys()

    # check to see if plot methods correctly configured
    if "plot_method" not in keys:
        plotting_entry["plot_method"] = []
    else:
        type_check(plotting_entry["plot_method"], list, "plot_method")
        list_type_check(plotting_entry["plot_method"], str, "plot_method")

        classif = ["conf_matrix", "roc_curve"]
        reg = ["hist_overlapped", "joint", "joint_dens", "corr"]
        both = ["barplot_scorer", "boxplot_scorer", "shap_plots", "permut_imp_test"]

        max_opts = set(both + (reg if problem == REGRESSION else classif))
        given_opts = set(plotting_entry["plot_method"])

        if not given_opts.issubset(max_opts):
            raise ValueError(
                f'Non-valid plot options given for plot_method: {",".join(list(given_opts-max_opts))}. Possible options: '
                f"{max_opts}"
            )

    # check to see if permut_imp_test is correctly configured
    if "permut_imp_test" in plotting_entry["plot_method"]:
        if "top_feats_permImp" not in keys:
            plotting_entry["top_feats_permImp"] = 20
        else:
            type_check(plotting_entry["top_feats_permImp"], int, "top_feats_permImp")
            if plotting_entry["top_feats_permImp"] <= 0:
                raise ValueError("top_feats_permImp must be greater than 0")

    # check to see if permut_imp_test is correctly configured
    if "shap_plots" in plotting_entry["plot_method"]:
        if "top_feats_shap" not in keys:
            plotting_entry["top_feats_shap"] = 20
        else:
            type_check(plotting_entry["top_feats_shap"], int, "top_feats_shap")
            if plotting_entry["top_feats_shap"] <= 0:
                raise ValueError("top_feats_shap must be greater than 0")

        if "explanations_data" not in keys:
            plotting_entry["explanations_data"] = "all"
        else:
            type_check(plotting_entry["explanations_data"], str, "explanations_data")
            if plotting_entry["explanations_data"] not in ["test", "exemplars", "all"]:
                raise ValueError(
                    'explanations_data must one of: "test", "exemplars", "all"'
                )

    return plotting_entry


#################### auto method passing ####################
def parse_autoxgboost(xgbEntry):
    keys = xgbEntry.keys()
    validKeys = {"verbose", "n_trials", "timeout"}
    if not set(keys).issubset(validKeys):
        raise ValueError(
            f"Invalid entry for autoxgboost_config: {set(keys)-validKeys}. Valid options: {validKeys}"
        )

    if "verbose" not in keys:
        xgbEntry["verbose"] = False
    else:
        type_check(xgbEntry["verbose"], bool, "autoxgboost_config:verbose")

    if "n_trials" not in keys:
        xgbEntry["n_trials"] = 10
    else:
        type_check(xgbEntry["n_trials"], int, "autoxgboost_config:n_trials")
        if xgbEntry["n_trials"] < 1:
            raise ValueError("autoxgboost:n_trials must be an int greater than 0")

    if "timeout" not in keys:
        xgbEntry["timeout"] = 1000
    else:
        type_check(xgbEntry["timeout"], int, "autoxgboost_config:timeout")
        if xgbEntry["timeout"] < 1:
            raise ValueError("autoxgboost:timeout must be an int greater than 0")
    return xgbEntry


def parse_autolgbm(lgbmEntry):
    keys = lgbmEntry.keys()
    validKeys = {"verbose", "n_trials", "timeout"}
    if not set(keys).issubset(validKeys):
        raise ValueError(
            f"Invalid entry for autolgbm_config: {set(keys)-validKeys}. Valid options: {validKeys}"
        )

    if "verbose" not in keys:
        lgbmEntry["verbose"] = False
    else:
        type_check(lgbmEntry["verbose"], bool, "autolgbm_config:verbose")

    if "n_trials" not in keys:
        lgbmEntry["n_trials"] = 10
    else:
        type_check(lgbmEntry["n_trials"], int, "autolgbm_config:n_trials")
        if lgbmEntry["n_trials"] < 1:
            raise ValueError("autolgbm:n_trials must be an int greater than 0")

    if "timeout" not in keys:
        lgbmEntry["timeout"] = 1000
    else:
        type_check(lgbmEntry["timeout"], int, "autolgbm_config:timeout")
        if lgbmEntry["timeout"] < 1:
            raise ValueError("autolgbm:timeout must be an int greater than 0")
    return lgbmEntry


def parse_autokeras(kerasEntry):
    keys = kerasEntry.keys()
    validKeys = {
        "n_epochs",
        "batch_size",
        "verbose",
        "n_blocks",
        "dropout",
        "use_batchnorm",
        "n_trials",
        "tuner",
    }

    if not set(keys).issubset(validKeys):
        raise ValueError(
            f"Invalid entry for autokeras_config: {set(keys)-validKeys}. Valid options: {validKeys}"
        )

    if "verbose" not in keys:
        kerasEntry["verbose"] = False
    else:
        type_check(kerasEntry["verbose"], bool, "autokeras_config:verbose")

    if "use_batchnorm" not in keys:
        kerasEntry["use_batchnorm"] = True
    else:
        type_check(kerasEntry["use_batchnorm"], bool, "autokeras_config:use_batchnorm")

    if "n_trials" not in keys:
        kerasEntry["n_trials"] = 4
    else:
        type_check(kerasEntry["n_trials"], int, "autokeras_config:n_trials")
        if kerasEntry["n_trials"] < 1:
            raise ValueError("autokeras:n_trials must be an int greater than 0")

    if "n_epochs" not in keys:
        kerasEntry["n_epochs"] = 100
    else:
        type_check(kerasEntry["n_epochs"], int, "autokeras_config:n_epochs")
        if kerasEntry["n_epochs"] < 1:
            raise ValueError("autokeras:n_epochs must be an int greater than 0")

    if "batch_size" not in keys:
        kerasEntry["batch_size"] = 32
    else:
        type_check(kerasEntry["batch_size"], int, "autokeras_config:batch_size")
        if kerasEntry["batch_size"] < 1:
            raise ValueError("autokeras:batch_size must be an int greater than 0")

    if "n_blocks" not in keys:
        kerasEntry["n_blocks"] = 3
    else:
        type_check(kerasEntry["n_blocks"], int, "autokeras_config:n_blocks")
        if kerasEntry["n_blocks"] < 1:
            raise ValueError("autokeras:n_blocks must be an int greater than 0")

    if "dropout" not in keys:
        kerasEntry["dropout"] = 0.3
    else:
        type_check(kerasEntry["dropout"], float, "autokeras_config:dropout")
        if (kerasEntry["dropout"] < 0) or (kerasEntry["dropout"] > 1):
            raise ValueError("autokeras:dropout must be an float between 0 & 1")

    if "tuner" not in keys:
        kerasEntry["tuner"] = "bayesian"
    else:
        type_check(kerasEntry["tuner"], str, "autokeras_config:tuner")
        valid = ["bayesian", "greedy", "hyperband", "random"]
        if kerasEntry["tuner"] not in valid:
            raise ValueError(
                f'autokeras:tuner not valid: {kerasEntry["tuner"]}. Valid options: {valid}'
            )

    return kerasEntry


#################### problem ####################
def parse_MLSettings(problemEntry):
    validKeys = {
        "seed_num",
        "test_size",
        "problem_type",
        "hyper_tuning",
        "hyper_budget",
        "stratify_by_groups",
        "groups",
        "balancing",
        "fit_scorer",
        "scorer_list",
        "model_list",
        "autokeras_config",
        "autolgbm_config",
        "autoxgboost_config",
        "encoding",
        "feature_selection",
    }
    keys = set(problemEntry.keys())

    if not keys.issubset(validKeys):
        raise ValueError(
            f"Invalid entry for problemEntry: {keys-validKeys}. Valid options: {validKeys}"
        )

    if "seed_num" not in keys:
        problemEntry["seed_num"] = 29292
    else:
        type_check(problemEntry["seed_num"], int, "seed_num")
        if problemEntry["seed_num"] < 0:
            raise ValueError("seed_num must be an int greater than 0")

    if "test_size" not in keys:
        problemEntry["test_size"] = 0.2
    else:
        type_check(problemEntry["test_size"], float, "test_size")
        if (problemEntry["test_size"] < 0) or (problemEntry["test_size"] > 1):
            raise ValueError("test_size must be a float between 0 and 1")

    ###################################### MANDITORY ######################################
    if "problem_type" not in keys:
        raise ValueError(
            "problem_type must be given. Options: CLASSIFICATION or REGRESSION"
        )
    else:
        type_check(problemEntry["problem_type"], str, "problem_type")
        if problemEntry["problem_type"] not in [CLASSIFICATION, REGRESSION]:
            raise ValueError(
                f'problem_type not valid: {problemEntry["problem_type"]}. Must be CLASSIFICATION or REGRESSION'
            )

    if "hyper_tuning" not in keys:
        problemEntry["hyper_tuning"] = "random"
    else:
        if problemEntry["hyper_tuning"] is not None:
            type_check(problemEntry["hyper_tuning"], str, "hyper_tuning")
            if problemEntry["hyper_tuning"] not in ["random", "grid"]:
                raise ValueError(
                    f'hyper_tuning option invalid: {problemEntry["hyper_tuning"]}. Must be "random" or "grid" or None'
                )

    if "hyper_budget" not in keys:
        if problemEntry["hyper_tuning"] == "random":
            problemEntry["hyper_budget"] = 50
        else:
            problemEntry["hyper_budget"] = None
    else:
        if problemEntry["hyper_tuning"] == "random":
            type_check(problemEntry["hyper_budget"], int, "hyper_budget")
            if problemEntry["hyper_budget"] < 0:
                raise ValueError("hyper_budget must be an int greater than 0")

    if "stratify_by_groups" not in keys:
        problemEntry["stratify_by_groups"] = "N"
    else:
        type_check(problemEntry["stratify_by_groups"], str, "stratify_by_groups")
        if problemEntry["stratify_by_groups"] not in ["Y", "N"]:
            raise ValueError('stratify_by_groups must be either "Y" or "N"')

    if "groups" not in keys:
        problemEntry["groups"] = None
    else:
        type_check(problemEntry["groups"], str, "groups")

    if "balancing" not in keys:
        problemEntry["balancing"] = "NONE"
    else:
        type_check(problemEntry["balancing"], str, "balancing")
        if problemEntry["balancing"] not in ["OVER", "UNDER", "NONE"]:
            raise ValueError('balancing must be either "OVER", "UNDER", "NONE"')

    if "fit_scorer" not in keys:
        problemEntry["fit_scorer"] = (
            "f1_score"
            if problemEntry["problem_type"] == CLASSIFICATION
            else "mean_absolute_percentage_error"
        )
    else:
        type_check(problemEntry["fit_scorer"], str, "fit_scorer")

        valid = (
            list(METRICS[CLASSIFICATION].keys())
            if problemEntry["problem_type"] == CLASSIFICATION
            else list(METRICS[REGRESSION].keys())
        )
        if problemEntry["fit_scorer"] not in valid:
            raise ValueError(
                f"fit_scorer must be one of: {valid}, provided: {problemEntry['fit_scorer']}"
            )

    if "scorer_list" not in keys:
        problemEntry["scorer_list"] = (
            list(METRICS[CLASSIFICATION].keys())
            if problemEntry["problem_type"] == CLASSIFICATION
            else list(METRICS[REGRESSION].keys())
        )
    else:
        type_check(problemEntry["scorer_list"], list, "scorer_list")
        list_type_check(problemEntry["scorer_list"], str, "scorer_list")

        if problemEntry["fit_scorer"] not in problemEntry["scorer_list"]:
            problemEntry["scorer_list"].append(problemEntry["fit_scorer"])

        given_opts = set(problemEntry["scorer_list"])
        metric_max_opts = set(
            METRICS[CLASSIFICATION].keys()
            if problemEntry["problem_type"] == CLASSIFICATION
            else METRICS[REGRESSION].keys()
        )

        if not given_opts.issubset(metric_max_opts):
            raise ValueError(
                f'Non-valid options given for scorer_list: {",".join(list(given_opts-metric_max_opts))}. Possible '
                f"options: {metric_max_opts}"
            )

    model_max_opts = set(
        list(MODELS[problemEntry["problem_type"]].keys()) + list(MODELS["both"].keys())
    )
    ###################################### MANDITORY ######################################
    if "model_list" not in keys:
        raise ValueError(
            f"model_list must be a list containg at least one option from: {model_max_opts}"
        )
    else:
        type_check(problemEntry["model_list"], list, "model_list")
        list_type_check(problemEntry["model_list"], str, "model_list")

        if problemEntry["model_list"] == []:
            raise ValueError(
                f"model_list must contain at least one option from: {model_max_opts}"
            )
        else:
            given_opts = set(problemEntry["model_list"])
            if not given_opts.issubset(model_max_opts):
                raise ValueError(
                    f'Non-valid options given for model_list: {",".join(list(given_opts-model_max_opts))}. Possible '
                    f"options: {model_max_opts}"
                )

    if problemEntry["problem_type"] == CLASSIFICATION:
        if "encoding" not in keys:
            problemEntry["encoding"] = None
        else:
            if problemEntry["encoding"] is not None:
                type_check(problemEntry["encoding"], str, "encoding")
                if problemEntry["encoding"] not in ["label", "onehot"]:
                    raise ValueError(
                        f'Encoding entry ({problemEntry["encoding"]}) not valid, must be None or "label" or "onehot"'
                    )

    if "AutoKeras" in problemEntry["model_list"]:
        autoEnt = problemEntry.get("autokeras_config")
        autoEnt = {} if autoEnt is None else autoEnt
        problemEntry["autokeras_config"] = parse_autokeras(autoEnt)

    if "AutoLGBM" in problemEntry["model_list"]:
        autoEnt = problemEntry.get("autolgbm_config")
        autoEnt = {} if autoEnt is None else autoEnt
        problemEntry["autolgbm_config"] = parse_autolgbm(autoEnt)

    if "AutoXGBoost" in problemEntry["model_list"]:
        autoEnt = problemEntry.get("autoxgboost_config")
        autoEnt = {} if autoEnt is None else autoEnt
        problemEntry["autoxgboost_config"] = parse_autoxgboost(autoEnt)

    if "feature_selection" not in keys:
        problemEntry["feature_selection"] = None
    elif problemEntry["feature_selection"] is not None:
        problemEntry["feature_selection"] = parse_FS_settings(
            problemEntry["problem_type"], problemEntry["feature_selection"]
        )

    return problemEntry


#################### data ####################
def parse_data(dataEntry):
    validKeys = {
        "name",
        "file_path",
        "metadata_file",
        "file_path_holdout_data",
        "metadata_file_holdout_data",
        "save_path",
        "target",
        "data_type",
    }
    keys = set(dataEntry.keys())

    if not keys.issubset(validKeys):
        raise ValueError(
            f"Invalid entry for dataEntry: {keys-validKeys}. Valid options: {validKeys}"
        )

    if (
        "name" not in keys
    ):  ###################################### MANDITORY ######################################
        raise ValueError("name must be given")
    else:
        type_check(dataEntry["name"], str, "name")

    if (
        "file_path" not in keys
    ):  ###################################### MANDITORY ######################################
        raise ValueError("file_path must be defined")
    else:
        type_check(dataEntry["file_path"], str, "file_path")
        if not exists(dataEntry["file_path"]):
            raise ValueError("File given in file_path does not exist")

    if "metadata_file" not in keys:
        # Set to an empty string so that it is assumed the main dataset given contains the target
        dataEntry["metadata_file"] = ""
    else:
        type_check(dataEntry["metadata_file"], str, "metadata_file")
        entry = dataEntry["metadata_file"]
        if entry and not exists(entry):
            raise ValueError("File given in metadata_file does not exist")

    if "file_path_holdout_data" not in keys:
        dataEntry["file_path_holdout_data"] = None
    else:
        type_check(dataEntry["file_path_holdout_data"], str, "file_path_holdout_data")
        if not exists(dataEntry["file_path_holdout_data"]):
            raise ValueError("File given in file_path_holdout_data does not exist")

    if "metadata_file_holdout_data" not in keys:
        dataEntry["metadata_file_holdout_data"] = None
    else:
        type_check(
            dataEntry["metadata_file_holdout_data"], str, "metadata_file_holdout_data"
        )
        if not exists(dataEntry["metadata_file_holdout_data"]):
            raise ValueError("File given in metadata_file_holdout_data does not exist")

    if "save_path" not in keys:
        dataEntry["save_path"] = "/experiments/"
    else:
        type_check(dataEntry["save_path"], str, "save_path")

    if (
        "target" not in keys
    ):  ###################################### MANDITORY ######################################
        raise ValueError("target must be defined")
    else:
        type_check(dataEntry["target"], str, "target")

    if (
        "data_type" not in keys
    ):  ###################################### MANDITORY ######################################
        raise ValueError("data_type must be defined")
    else:
        type_check(dataEntry["data_type"], str, "data_type")
        valid = ["tabular", "gene_expression", "microbiome", "metabolomic", "other"]
        if dataEntry["data_type"] not in valid:
            raise ValueError(
                f'data_type option not valid: {dataEntry["data_type"]}. Valid options: {valid}'
            )

    return dataEntry


#################### omic parsing ####################
def parse_microbiome(omicEntry):
    validKeys = {
        "collapse_tax",
        "min_reads",
        "norm_reads",
        "filter_abundance",
        "filter_prevalence",
        "filter_microbiome_samples",
        "remove_classes",
        "merge_classes",
    }
    keys = set(omicEntry.keys())

    if not keys.issubset(validKeys):
        raise ValueError(
            f"Invalid entry for dataEntry: {keys-validKeys}. Valid options: {validKeys}"
        )

    if "collapse_tax" not in keys:
        omicEntry["collapse_tax"] = None
    else:
        if omicEntry["collapse_tax"] is not None:
            type_check(omicEntry["collapse_tax"], str, "collapse_tax")
            if omicEntry["collapse_tax"] not in ["genus", "species"]:
                raise ValueError(
                    f'microbiome:collapse_tax option not valid: {omicEntry["collapse_tax"]}. Valid options: '
                    f'{["genus", "species"]}'
                )

    if "min_reads" not in keys:
        omicEntry["min_reads"] = None
    else:
        if omicEntry["min_reads"] is not None:
            type_check(omicEntry["min_reads"], int, "min_reads")
            if omicEntry["min_reads"] < 0:
                raise ValueError(
                    "microbiome:min_reads entry must be an int greater than 0"
                )

    if "norm_reads" not in keys:
        omicEntry["norm_reads"] = None
    else:
        if omicEntry["norm_reads"] is not None:
            type_check(omicEntry["norm_reads"], int, "norm_reads")
            if omicEntry["norm_reads"] < 0:
                raise ValueError(
                    "microbiome:norm_reads entry must be an int greater than 0"
                )

    if "filter_abundance" not in keys:
        omicEntry["filter_abundance"] = None
    else:
        if omicEntry["filter_abundance"] is not None:
            type_check(omicEntry["filter_abundance"], int, "filter_abundance")
            if omicEntry["filter_abundance"] < 0:
                raise ValueError(
                    "microbiome:filter_abundance entry must be an int greater than 0"
                )

    if "filter_prevalence" not in keys:
        omicEntry["filter_prevalence"] = None
    else:
        if omicEntry["filter_prevalence"] is not None:
            type_check(omicEntry["filter_prevalence"], float, "filter_prevalence")
            if (omicEntry["filter_prevalence"] < 0) or (
                omicEntry["filter_prevalence"] > 1
            ):
                raise ValueError(
                    "microbiome:filter_prevalence must be a float between 0 and 1"
                )

    if "filter_microbiome_samples" not in keys:
        omicEntry["filter_microbiome_samples"] = None
    else:
        if omicEntry["filter_microbiome_samples"] is not None:
            pass

    if "remove_classes" not in keys:
        omicEntry["remove_classes"] = None
    else:
        if omicEntry["remove_classes"] is not None:
            type_check(omicEntry["remove_classes"], list, "microbiome:remove_classes")
            list_type_check(
                omicEntry["remove_classes"], str, "microbiome:remove_classes"
            )

    if "merge_classes" not in keys:
        omicEntry["merge_classes"] = None
    else:
        if omicEntry["merge_classes"] is not None:
            type_check(omicEntry["merge_classes"], dict, "merge_classes")
            for key in omicEntry["merge_classes"].keys():
                type_check(
                    omicEntry["merge_classes"][key],
                    list,
                    f"microbiome:merge_classes:{key}",
                )
                list_type_check(
                    omicEntry["merge_classes"][key],
                    str,
                    f"microbiome:merge_classes:{key}",
                )
    return omicEntry


def parse_geneExpression(omicEntry):
    validKeys = {
        "expression_type",
        "filter_sample",
        "filter_genes",
        "output_file_ge",
        "output_metadata",
    }
    keys = set(omicEntry.keys())

    if not keys.issubset(validKeys):
        raise ValueError(
            f"Invalid entry for dataEntry: {keys-validKeys}. Valid options: {validKeys}"
        )

    if "expression_type" not in keys:
        raise ValueError(
            'expression_type must be given and one of: "FPKM", "RPKM", "TMM", "TPM", "Log2FC", "COUNTS", "OTHER"'
        )
    else:
        if omicEntry["expression_type"] is not None:
            type_check(omicEntry["expression_type"], str, "expression_type")
            valid = ["FPKM", "RPKM", "TMM", "TPM", "Log2FC", "COUNTS", "OTHER"]
            if omicEntry["expression_type"] not in valid:
                raise ValueError(
                    f'geneExpression:expression_type option not valid: {omicEntry["expression_type"]}. Valid options: '
                    f"{valid}"
                )

    if "filter_sample" not in keys:
        omicEntry["filter_sample"] = 0
    else:
        if omicEntry["filter_sample"] is not None:
            try:
                type_check(omicEntry["filter_sample"], int, "filter_sample")
            except Exception:
                try:
                    type_check(omicEntry["filter_sample"], float, "filter_sample")
                except Exception:
                    raise ValueError("filter_sample must be a int or a float >0")
            if omicEntry["filter_sample"] < 0:
                raise ValueError("filter_sample must be greater than 0.")

    if "filter_genes" not in keys:
        omicEntry["filter_genes"] = [0, 0]
    else:
        if omicEntry["filter_genes"] is not None:
            type_check(omicEntry["filter_genes"], list, "filter_genes")
            list_type_check(omicEntry["filter_genes"], int, "filter_genes")
            if len(omicEntry["filter_genes"]) != 2:
                raise ValueError(
                    "filter_genes must be a list of 2 integers greater than 0."
                )
            if (omicEntry["filter_genes"][0] < 0) or (omicEntry["filter_genes"][1] < 0):
                raise ValueError(
                    "filter_genes must be a list of 2 integers greater than 0."
                )

    if "output_file_ge" not in keys:
        omicEntry["output_file_ge"] = None
    else:
        if omicEntry["output_file_ge"] is not None:
            type_check(omicEntry["output_file_ge"], str, "output_file_ge")

    if "output_metadata" not in keys:
        omicEntry["output_metadata"] = None
    else:
        if omicEntry["output_metadata"] is not None:
            type_check(omicEntry["output_metadata"], str, "output_metadata")

    return omicEntry


def parse_metabolomic(omicEntry):
    validKeys = {
        "filter_metabolomic_sample",
        "filter_measurements",
        "output_file_met",
        "output_metadata",
    }
    keys = set(omicEntry.keys())

    if not keys.issubset(validKeys):
        raise ValueError(
            f"Invalid entry for dataEntry: {keys-validKeys}. Valid options: {validKeys}"
        )

    if "filter_metabolomic_sample" not in keys:
        omicEntry["filter_metabolomic_sample"] = 0
    else:
        if omicEntry["filter_metabolomic_sample"] is not None:
            try:
                type_check(
                    omicEntry["filter_metabolomic_sample"],
                    int,
                    "filter_metabolomic_sample",
                )
            except Exception:
                try:
                    type_check(
                        omicEntry["filter_metabolomic_sample"],
                        float,
                        "filter_metabolomic_sample",
                    )
                except Exception:
                    raise ValueError(
                        "filter_metabolomic_sample must be a int or a float >0"
                    )
            if omicEntry["filter_metabolomic_sample"] < 0:
                raise ValueError("filter_metabolomic_sample must be greater than 0.")

    if "filter_measurements" not in keys:
        omicEntry["filter_measurements"] = [0, 0]
    else:
        if omicEntry["filter_measurements"] is not None:
            type_check(omicEntry["filter_measurements"], list, "filter_measurements")
            list_type_check(
                omicEntry["filter_measurements"], int, "filter_measurements"
            )
            if len(omicEntry["filter_measurements"]) != 2:
                raise ValueError(
                    "filter_measurements must be a list of 2 integers greater than 0."
                )
            if (omicEntry["filter_measurements"][0] < 0) or (
                omicEntry["filter_measurements"][1] < 0
            ):
                raise ValueError(
                    "filter_measurements must be a list of 2 integers greater than 0."
                )

    if "output_file_met" not in keys:
        omicEntry["output_file_met"] = None
    else:
        if omicEntry["output_file_met"] is not None:
            type_check(omicEntry["output_file_met"], str, "output_file_met")

    if "output_metadata" not in keys:
        omicEntry["output_metadata"] = None
    else:
        if omicEntry["output_metadata"] is not None:
            type_check(omicEntry["output_metadata"], str, "output_metadata")

    return omicEntry


def parse_tabular(omicEntry):
    validKeys = {
        "filter_tabular_sample",
        "filter_tabular_measurements",
        "output_file_tab",
        "output_metadata",
    }
    keys = set(omicEntry.keys())

    if not keys.issubset(validKeys):
        raise ValueError(
            f"Invalid entry for dataEntry: {keys-validKeys}. Valid options: {validKeys}"
        )

    if "filter_tabular_sample" not in keys:
        omicEntry["filter_tabular_sample"] = 0
    else:
        if omicEntry["filter_tabular_sample"] is not None:
            try:
                type_check(
                    omicEntry["filter_tabular_sample"], int, "filter_tabular_sample"
                )
            except Exception:
                try:
                    type_check(
                        omicEntry["filter_tabular_sample"],
                        float,
                        "filter_tabular_sample",
                    )
                except Exception:
                    raise ValueError(
                        "filter_tabular_sample must be a int or a float >0"
                    )
            if omicEntry["filter_tabular_sample"] < 0:
                raise ValueError(
                    "filter_tabular_sample must be greater than or equal to 0."
                )

    if "filter_tabular_measurements" not in keys:
        omicEntry["filter_tabular_measurements"] = [0, 0]
    else:
        if omicEntry["filter_tabular_measurements"] is not None:
            type_check(
                omicEntry["filter_tabular_measurements"],
                list,
                "filter_tabular_measurements",
            )
            list_type_check(
                omicEntry["filter_tabular_measurements"],
                int,
                "filter_tabular_measurements",
            )
            if len(omicEntry["filter_tabular_measurements"]) != 2:
                raise ValueError(
                    "filter_tabular_measurements must be a list of 2 integers greater than 0."
                )
            if (omicEntry["filter_tabular_measurements"][0] < 0) or (
                omicEntry["filter_tabular_measurements"][1] < 0
            ):
                raise ValueError(
                    "filter_tabular_measurements must be a list of 2 integers greater than 0."
                )

    if "output_file_tab" not in keys:
        omicEntry["output_file_tab"] = None
    else:
        if omicEntry["output_file_tab"] is not None:
            type_check(omicEntry["output_file_tab"], str, "output_file_tab")

    if "output_metadata" not in keys:
        omicEntry["output_metadata"] = None
    else:
        if omicEntry["output_metadata"] is not None:
            type_check(omicEntry["output_metadata"], str, "output_metadata")

    return omicEntry


def parse_omic(config):
    if config["data"]["data_type"] != "other":
        if config["data"]["data_type"] == "tabular":
            omicEnt = config.get("tabular")
            omicEnt = {} if omicEnt is None else omicEnt
            config["tabular"] = parse_tabular(omicEnt)
        if config["data"]["data_type"] == "gene_expression":
            omicEnt = config.get("gene_expression")
            omicEnt = {} if omicEnt is None else omicEnt
            config["gene_expression"] = parse_geneExpression(omicEnt)
        if config["data"]["data_type"] == "microbiome":
            omicEnt = config.get("microbiome")
            omicEnt = {} if omicEnt is None else omicEnt
            config["microbiome"] = parse_microbiome(omicEnt)
        if config["data"]["data_type"] == "metabolomic":
            omicEnt = config.get("metabolomic")
            omicEnt = {} if omicEnt is None else omicEnt
            config["metabolomic"] = parse_metabolomic(omicEnt)
    return config


#################### prediction ####################


def parse_prediction(predictionEntry):
    validKeys = {"file_path", "outfile_name", "metadata_file"}
    keys = set(predictionEntry.keys())

    if not keys.issubset(validKeys):
        raise ValueError(
            f"Invalid entry for predictionEntry: {keys-validKeys}. Valid options: {validKeys}"
        )

    if (
        "file_path" not in keys
    ):  ###################################### MANDITORY ######################################
        raise ValueError("prediction:file_path must be defined")
    else:
        type_check(predictionEntry["file_path"], str, "file_path")
        if not exists(predictionEntry["file_path"]):
            raise ValueError("File given in prediction:file_path does not exist")

    if "metadata_file" not in keys:
        predictionEntry["metadata_file"] = None

    if (
        ("outfile_name" not in keys)
        or (predictionEntry["outfile_name"] == "")
        or (predictionEntry["outfile_name"] is None)
    ):
        predictionEntry["outfile_name"] = "prediction_results"
    else:
        type_check(predictionEntry["outfile_name"], str, "outfile_name")

    return predictionEntry


#################### config ####################
def parse_config(config):
    config["data"] = parse_data(config["data"])
    config["ml"] = parse_MLSettings(config["ml"])
    pltEnt = config.get("plotting")
    pltEnt = {} if pltEnt is None else pltEnt
    config["plotting"] = parser_plotting(pltEnt, config["ml"]["problem_type"])
    config = parse_omic(config)

    if "prediction" in config.keys():
        config["prediction"] = parse_prediction(config["prediction"])

    return config


#################### feature selection ####################
def validate_FS_models_and_metrics(problem_type, estimator, metric):
    """
    Check if given the problem type that the estimator and evaluation metric chosen is valid or not
    """
    omicLogger.debug("Validating model and metric settings...")
    # check that the estimator is loaded in
    if estimator not in MODELS[problem_type].keys():
        raise ValueError(f"{estimator} is not currently available for use")
    else:
        est = MODELS[problem_type][estimator]["model"]

    # check that the metric is loaded in
    if not (
        metric in METRICS[REGRESSION].keys() or metric in METRICS[CLASSIFICATION].keys()
    ):
        raise ValueError(f"{metric} is not currently available for use")

    # check that the estimator selected is appropriate for the problem type
    if not (
        ((problem_type == REGRESSION) and (est._estimator_type == "regressor"))
        or ((problem_type == CLASSIFICATION) and (est._estimator_type == "classifier"))
    ):
        raise ValueError(
            f"{estimator} is not a valid method for a {problem_type} problem"
        )

    # check that the metric selected is appropriate for the problem types
    if metric not in METRICS[problem_type].keys():
        raise ValueError(f"{metric} is not a valid method for a {problem_type} problem")

    return METRICS[problem_type][metric]._sign == -1


def parse_FS_model_inputs(problem_type, eval_model, eval_metric):
    omicLogger.debug("Parsing model inputs...")
    # check we have a valid problem type
    if not ((problem_type == CLASSIFICATION) or (problem_type == REGRESSION)):
        raise ValueError("PROBLEM TYPE IS NOT CLASSIFICATION OR REGRESSION")

    # set the evaluation model and metric if we have been given None
    if eval_model is None:
        eval_model = (
            "RandomForestClassifier"
            if problem_type == CLASSIFICATION
            else "RandomForestRegressor"
        )

    if eval_metric is None:
        eval_metric = (
            "f1_score" if problem_type == CLASSIFICATION else "mean_squared_error"
        )

    # check the combination of model and metric is valid
    low = validate_FS_models_and_metrics(problem_type, eval_model, eval_metric)

    return eval_model, eval_metric, low


def parse_FS_settings(problem_type, FS_dict):
    """
    A function to check ALL the FS setting to ensure correct/valid entiries/combinations
    """
    omicLogger.debug("Parsing feature selection settings...")

    keys = FS_dict.keys()

    if "k" in keys:
        k = FS_dict["k"]
    else:
        k = "auto"

    if "var_threshold" in keys:
        threshold = FS_dict["var_threshold"]
    else:
        threshold = 0

    if "method" in keys:
        method_dict = FS_dict["method"]

        if method_dict["name"] not in FS_METHODS.keys():
            raise ValueError(
                f"{method_dict['name']} not currently available for use. please select a different method."
            )

        elif method_dict["name"] == "SelectKBest":
            if ("metric" not in method_dict.keys()) or (method_dict["metric"] is None):
                method_dict["metric"] = (
                    "f_classif" if problem_type == CLASSIFICATION else "f_regression"
                )

            elif method_dict["metric"] not in FS_KBEST_METRICS.keys():
                raise ValueError(
                    f"{method_dict['metric']} not currently available for use. please select a different metric."
                )

            else:
                metrics_reg = ["f_regression", "mutual_info_regression"]
                metrics_clf = ["f_classif", "mutual_info_classif"]

                if (
                    (problem_type == CLASSIFICATION)
                    and (method_dict["metric"] not in metrics_clf)
                ) or (
                    (problem_type == REGRESSION)
                    and (method_dict["metric"] not in metrics_reg)
                ):
                    raise ValueError(
                        f"{method_dict['metric']} is not appropriate for problem type {problem_type}."
                    )

        elif method_dict["name"] == "RFE":
            if ("estimator" not in method_dict.keys()) or (
                method_dict["estimator"] is None
            ):
                method_dict["estimator"] = (
                    "RandomForestClassifier"
                    if problem_type == CLASSIFICATION
                    else "RandomForestRegressor"
                )
            elif method_dict["estimator"] not in MODELS[problem_type].keys():
                raise ValueError(
                    f"{method_dict['estimator']} is not appropriate for problem type {problem_type}."
                )
    else:
        method_dict = {
            "name": "SelectKBest",
            "metric": "f_classif" if problem_type == CLASSIFICATION else "f_regression",
        }

    if "auto" in keys:
        auto_dict = FS_dict["auto"]
        if "min_features" not in auto_dict.keys():
            auto_dict["min_features"] = 10

        if "max_features" not in auto_dict.keys():
            auto_dict["max_features"] = None

        if "interval" not in auto_dict.keys():
            auto_dict["interval"] = 1

        if "eval_model" not in auto_dict.keys():
            auto_dict["eval_model"] = None

        if "eval_metric" not in auto_dict.keys():
            auto_dict["eval_metric"] = None

        (
            auto_dict["eval_model"],
            auto_dict["eval_metric"],
            auto_dict["low"],
        ) = parse_FS_model_inputs(
            problem_type, auto_dict["eval_model"], auto_dict["eval_metric"]
        )
    else:
        auto_dict = {
            "min_features": 10,
            "max_features": None,
            "interval": 1,
            "eval_model": (
                "RandomForestClassifier"
                if problem_type == CLASSIFICATION
                else "RandomForestRegressor"
            ),
            "eval_metric": (
                "f1_score" if problem_type == CLASSIFICATION else "mean_squared_error"
            ),
            "low": problem_type != CLASSIFICATION,
        }

    if method_dict["name"] == "RFE":
        auto_dict["eval_model"] = method_dict["estimator"]

    fs_dict = {
        "k": k,
        "var_threshold": threshold,
        "method": method_dict,
        "auto": auto_dict,
    }

    return fs_dict
