#!/bin/bash
#
##############################
# Copyright IBM Corp. 2023
##############################
#
# @author James Strudwick IBM Research
#
#set -x # switch on
set +x # switch off

echo "This is the script to build Auto-Omics Docker Images"

CWD="$(basename $(pwd))"

if [[ -z $IMAGE_NAME ]]; then 
   echo "No Image Name defined in commit message."
   # return 1 - used with source 
   exit 1
fi

if ["$TRAVIS_BRANCH" = "DEV"];
then
    _imageTag="DEV"
else
    source _version.py
    _imageTag=${__version__}
fi
echo "Image Tag: $_imageTag"

echo "Note: Images built via this script are only pushed to the docker repository within this travis image."

#install buildx
docker buildx install

# build docker image
docker build --platform linux/amd64,linux/arm64 --no-cache --progress plain --tag $IMAGE_NAME:$_imageTag .
if [ $? -ne 0 ]; then
   echo "Failed to build image."
   exit 1
fi

echo "Finished !!"

############################################################################
# end script                                                               #
############################################################################