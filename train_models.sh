#! /usr/bin/env bash

. ./common.sh &&
docker run \
  --rm \
  -ti \
  -v ${PWD}/configs:/configs \
  -v ${PWD}/data:/data \
  ${NAME}:${VERSION} \
    python train_models.py -c /configs/L1000_exp.json