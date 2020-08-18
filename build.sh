#! /usr/bin/env bash

. ./common.sh &&
docker build --no-cache -t ${NAME}:${VERSION} . &&
docker tag ${NAME}:${VERSION} ${NAME}:latest &&
docker system prune -f