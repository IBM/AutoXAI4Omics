#! /usr/bin/env bash

. ./common.sh &&
docker run \
  --rm \
  -ti \
  -u ${USER_ID}:${GROUP_ID} \
  -v "${PWD}"/configs:/configs \
  -v "${PWD}"/data:/data \
  -v "${PWD}"/experiments:/experiments \
  ${NAME}:${VERSION} \
    bash
