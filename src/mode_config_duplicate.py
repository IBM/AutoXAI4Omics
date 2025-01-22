# Copyright (c) 2024 IBM Corp.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from argparse import ArgumentParser
from copy import deepcopy
import json
from pathlib import Path
from utils.parser.config_model import ConfigModel
from utils.load import load_config


def main():
    # TODO: 1. code comments
    # TODO: 2. code logic
    # TODO: 3. sensible var names
    # TODO: 4. type hint
    # TODO: 5. input validation
    # TODO: 6. docstring
    # TODO: 7. write test

    # get variables from CLI
    parser = ArgumentParser(description="config duplicator")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Template config to use for other data files",
    )
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        help="directory containing data files to have an associated config",
    )

    args = parser.parse_args()
    config_path = Path("/configs") / args.config
    if not (config_path.is_file() and config_path.exists()):
        raise ValueError(
            f"Value for -c/--config is not a file or does not exist within the `config/` directory. Path checked: {config_path}"
        )

    # get template config & validate
    config_model = ConfigModel(**load_config(config_path))
    out_config_dir = config_path.parent / "replicates"
    out_config_dir.mkdir(exist_ok=True)

    # directory containing data
    data_path = Path("/data") / args.dir
    # Validate data dir
    if not (data_path.is_dir() and data_path.exists()):
        raise ValueError(
            "Values for -d/-dir is not a directory or does not exist withing the `data/` directory"
        )

    # get all data paths
    data_files = list(data_path.rglob("*.csv"))

    # for each data file:
    for i, file in enumerate(data_files):
        print(
            f"Processing file {i:>0{len(str(len(data_files)))}}/{len(data_files)}",
            end="\r",
        )
        tmp_config = deepcopy(config_model)
        exp_name = "_".join(str(file.relative_to(data_path))[:-4].split("/"))
        tmp_config.data.name = exp_name
        tmp_config.data.file_path = file
        tmp_config.data.file_path_holdout_data = None
        tmp_config.data.metadata_file = None
        tmp_config.data.metadata_file_holdout_data = None
        # insert it into a file
        with open(str(out_config_dir / f"{exp_name}.json"), "w") as out_file:
            json.dump(json.loads(tmp_config.model_dump_json()), out_file, indent=4)

    print("Duplication completed" + " " * 50)


if __name__ == "__main__":
    main()
