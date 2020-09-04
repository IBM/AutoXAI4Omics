#! /usr/bin/env bash

. ./common.sh &&
docker run \
  --rm \
  -ti \
  -u aotuser \
  -m 8000000000 \
  -v "${PWD}"/configs:/configs \
  -v "${PWD}"/data:/data \
  -v "${PWD}"/experiments:/experiments \
  ${NAME}:${VERSION} \
    bash
