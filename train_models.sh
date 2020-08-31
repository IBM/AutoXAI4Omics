#! /usr/bin/env bash

. ./common.sh &&
docker run \
  --rm \
  -ti \
  -v "${PWD}"/configs:/configs \
  -v "${PWD}"/data:/data \
  -v "${PWD}"/experiments:/experiments \
  ${NAME}:${VERSION} \
    python train_models.py -c /configs/"$1"
