#! /usr/bin/env bash

. ./common_old.sh &&
docker run \
  --rm \
  -ti \
  -u ${USER_ID} \
  -v "${PWD}"/configs:/configs \
  -v "${PWD}"/data:/data \
  -v "${PWD}"/experiments:/experiments \
  ${NAME}:${VERSION} \
    bash
