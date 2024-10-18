#! /usr/bin/env bash
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

set -x

ROOT=''
DETACH=''
GPU=''
CONFIG=''
MODE=''
VOL_MAPS="-v ${PWD}/configs:/configs -v ${PWD}/data:/data -v ${PWD}/experiments:/experiments"

echo "Getting flags"
#get variables from input
while getopts 'm:c:rgd' OPTION; do
    case "$OPTION" in
        m) 
            case "${OPTARG}" in
                "train")
                    MODE=mode_train_models.py
                    ;;
                "test")
                    MODE=mode_testing_holdout.py
                    ;;
                "predict")
                    MODE=mode_predict.py
                    ;;
                "plotting")
                    MODE=mode_plotting.py
                    ;;
                "feature")
                    MODE=mode_feature_selection.py
                    ;;
                "bash")
                    MODE=bash
                    ;;
                ?)
                    echo "Unrecognised mode: ${OPTARG}. Valid modes: train, test, predict, plotting, bash"
                    exit 1
                    ;;
            esac
            ;;
        c)
            echo "Registering config"
            CONFIG="configs/${OPTARG}"
            if [ ! -d "$CONFIG" ] && [ ! -f "$CONFIG" ]
            then
                echo "config provided in -c flag (${CONFIG#configs/}) is not a valid directory or file"
                exit 1
            fi
            ;;
        r)
            echo "Setting root for bash"
            ROOT='-u root'
            ;;
        g)
            echo "Setting gpus access"
            GPU='--gpus all -ti'
            ;;
        d)
            echo "Registering container detachment"
            DETACH='-d'
            ;;
        ?)
          echo "script usage: $(basename \$0) [-m] [-c] [-p] [-r] [-g] [-d]" >&2
          exit 1
          ;;
    esac
done

if [[ $MODE == "" ]]
then
    echo "Please specify a mode using the flag -m, valid modes: train, test, predict, plotting, bash"
    exit 1
fi

if [[ $CONFIG == "" && $MODE != "bash" ]]
then
    echo "Config file not provided. please provide a file from the config folder"
    exit 1
fi

. ./common.sh
IMAGE_ID=$(docker images -q $IMAGE_FULL)

if [[ -z $IMAGE_ID ]]
then 
    echo "Image not built, please build by running './build.sh' "
    exit 1
fi

if [ $MODE != "bash" ]
then
    if [ -d "$CONFIG" ]
    echo "Entering batch mode..."
    then
        for FILE in $(find $CONFIG -name "*.json")
        do
            echo "Runing config file: $FILE"
            docker run \
              --rm \
              $DETACH \
              $GPU \
              $VOL_MAPS \
              $IMAGE_FULL \
              python $MODE -c /"$FILE"
        done
    else
        docker run \
          --rm \
          $DETACH \
          $GPU \
          $VOL_MAPS \
          $IMAGE_FULL \
          python $MODE -c /"$CONFIG"
    fi
else
    docker run \
      --rm \
      -ti \
      $ROOT \
      $VOL_MAPS \
      $IMAGE_FULL \
      bash
fi