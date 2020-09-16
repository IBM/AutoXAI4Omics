#! /usr/bin/env bash

if test -z "$1"; then
  echo "Missing the name of the configuration file to use"
  exit 1
fi

. ./common.sh &&
docker run \
  --rm \
  -ti \
  -u ${USER_ID} \
  -v "${PWD}"/configs:/configs \
  -v "${PWD}"/data:/data \
  -v "${PWD}"/experiments:/experiments \
  ${NAME}:${VERSION} \
    python AoT_gene_expression_pre_processing.py --expressionfile /data/"$1" --expressiontype OTHER --Filtersamples 100 --Filtergenes 0 0 --output /data/Processed_GE_L1000_discrete_drug_perturbation_blood_urea_nitrogen.csv
