#! /usr/bin/env bash

no_cache="--no-cache"
if test -z "${USE_NO_CACHE}"; then
  no_cache=""
fi

. ./common.sh &&
docker build ${no_cache} --build-arg USER_ID=${USER_ID} --build-arg GROUP_ID=${GROUP_ID} -t ${NAME}:${VERSION} . &&
docker tag ${NAME}:${VERSION} ${NAME}:latest &&
docker system prune -f
