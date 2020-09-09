#! /usr/bin/env bash

while getopts "r" opt; do
    case ${opt} in
        r)
            no_cache=--no-cache
            ;;
        *)
            no_cache=""
            ;;
    esac
done

. ./common.sh &&
docker build ${no_cache} --build-arg USER_ID=${USER_ID} --build-arg GROUP_ID=${GROUP_ID} -t ${NAME}:${VERSION} . &&
docker tag ${NAME}:${VERSION} ${NAME}:latest &&
docker system prune -f
