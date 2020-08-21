#! /usr/bin/env bash

. ./common.sh &&
docker build -t ${NAME}:${VERSION} . &&
docker tag ${NAME}:${VERSION} ${NAME}:latest &&
docker system prune -f