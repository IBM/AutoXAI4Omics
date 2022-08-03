#! /usr/bin/env bash

if test -z "$1"; then
  echo "Missing the name of the configuration file to use"
  exit 1
fi

pathname=`dirname "$1"`
configname=`basename "$1"`

. ./common.sh &&
docker run \
  --rm \
  -u ${USER_ID} \
  -v "${PWD}"/configs:/configs \
  -v "${PWD}"/data:/data \
  -v "${PWD}"/experiments:/experiments \
  ${NAME}:${VERSION} \
    python plotting.py -c /configs/"$configname"
