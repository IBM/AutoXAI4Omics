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
  docker build ${no_cache} -t ${IMAGE_NAME}:${IMAGE_TAG} . &&
  docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest &&
  docker system prune -f
