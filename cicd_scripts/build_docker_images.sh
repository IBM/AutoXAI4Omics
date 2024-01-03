#!/bin/bash
#
##############################
# Copyright IBM Corp. 2023
##############################
#
# @author James Strudwick IBM Research
#
set -x # switch on
# set +x # switch off

echo "This is the script to build & push OmiXai Docker Images"

echo "IBM Cloud Region: $IBM_CLOUD_REGION"
echo "Container Registry Region: $REGISTRY_REGION"
echo "Container Registry Namespace: $REGISTRY_NAMESPACE"
echo 
echo "Branch: $TRAVIS_BRANCH"
echo "Commit message: $TRAVIS_COMMIT_MESSAGE"
echo "Image name: $IMAGE_NAME"

if [[ "${TRAVIS_BRANCH}" == "DEV" ]];
then
    IMAGE_TAG="DEV"
else
    source _version.py
    IMAGE_TAG=${__version__}
fi
echo "Image Tag: $IMAGE_TAG"

############################################################################
# Download and install a few CLI tools and the Kubernetes Service plug-in. #
# Documentation on details can be found here:                              #
#    https://github.com/IBM-Cloud/ibm-cloud-developer-tools                #
############################################################################
echo "Install IBM Cloud CLI"
curl -sL https://ibm.biz/idt-installer | bash

############################################################################
# Log into the IBM Cloud environment using apikey                          #
############################################################################
echo "Login to IBM Cloud using apikey"
ibmcloud login -r ${IBM_CLOUD_REGION} --apikey ${IBM_CLOUD_API_KEY}
if [ $? -ne 0 ]; then
  echo "Failed to authenticate to IBM Cloud"
  exit 1
fi

############################################################################
# Set the right Region for IBM Cloud container registry                    #
############################################################################
echo "Switch to the correct region for the required IBM Cloud container registry"
ibmcloud cr region-set ${REGISTRY_REGION}
if [ $? -ne 0 ]; then
  echo "Failed to switch to correct IBM Cloud container registry region"
  exit 1
fi
############################################################################
# Log into the IBM Cloud container registry                                #
############################################################################
echo "Checking connected to IBM's Cloud container registry"
ibmcloud cr login
if [ $? -ne 0 ]; then
  echo "Failed to login to IBM's Cloud container registry"
  exit 1
fi
ibmcloud cr images

############################################################################
# configure docker buildx                                                  #
############################################################################
#install buildx
docker buildx install

#create builder
docker buildx create --platform=linux/arm64,linux/amd64 --use

#list builders for logging
docker buildx ls

############################################################################
# configure docker buildx                                                  #
############################################################################
# NOTE docker buildx automaticaly pushes to the repo

if [ $TRAVIS_BRANCH == "main" ]; then
    docker build --platform=linux/amd64,linux/arm64 \
    --no-cache --progress plain \
    --build-arg USER_ID=501 \
    --push --tag $REGISTRY_REGION/$REGISTRY_NAMESPACE/$IMAGE_NAME:$IMAGE_TAG \
    --tag $REGISTRY_REGION/$REGISTRY_NAMESPACE/$IMAGE_NAME:latest \
     \
    .
else
    docker build --platform=linux/amd64,linux/arm64 \
    --no-cache --progress plain \
    --build-arg USER_ID=501 \
    --push --tag $REGISTRY_REGION/$REGISTRY_NAMESPACE/$IMAGE_NAME:$IMAGE_TAG \
    .
fi

if [ $? -ne 0 ]; then
   echo "Failed to build image."
   exit 1
fi

echo "Finished !!"

############################################################################
# end script                                                               #
############################################################################