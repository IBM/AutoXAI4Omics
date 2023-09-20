#! /usr/bin/env bash
set -x

ROOT=''
GPU=''
CONFIG=''
MODE=''
VOL_MAPS="-v ${PWD}/configs:/configs -v ${PWD}/data:/data -v ${PWD}/experiments:/experiments"

echo "Getting flags"
#get variables from input
while getopts 'm:c:rg' OPTION; do
    case "$OPTION" in
        m) 
            case "${OPTARG}" in
                "train")
                    MODE=train_models.py
                    ;;
                "test")
                    MODE=testing_holdout.py
                    ;;
                "predict")
                    MODE=predict.py
                    ;;
                "plotting")
                    MODE=plotting.py
                    ;;
                "bash")
                    MODE=bash
                    ;;
                ?)
                    echo "Unrecognised mode: ${OPTARG}. Valid modes: train, test, predict, plotting, bash"
                    exit 1
                    ;;
        c)
            echo "Registering config"
            CONFIG=${OPTARG}
            ;;
        r)
            echo "Setting root for bash"
            ROOT='-u root'
            ;;
        g)
            echo "Setting gpus access"
            GPU='--gpus all -ti'
            ;;
        ?)
          echo "script usage: $(basename \$0) [-m] [-c] [-p] [-r] [-g]" >&2
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
    docker run \
      --rm \
      # -u ${USER_ID} \
      $GPU \
      $VOL_MAPS \
      $IMAGE_FULL \
      python $MODE -c /configs/"$CONFIG"
else
    docker run \
      --rm \
      -ti \
      # -u ${USER_ID} \
      $ROOT \
      $VOL_MAPS \
      $IMAGE_FULL \
      bash
fi