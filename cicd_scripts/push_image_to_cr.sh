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

echo "Script to push an image built within Travis-ci to IBM's Cloud image repo."
echo "This script is called by Travis-ci during the deploy stage."

echo "IBM Cloud Region: $IBM_CLOUD_REGION"
echo "Container Registry Region: $REGISTRY_REGION"
echo "Container Registry Namespace: $REGISTRY_NAMESPACE"
echo 
echo "Branch: $TRAVIS_BRANCH"
echo "Commit message: $TRAVIS_COMMIT_MESSAGE"
echo "Image name: $IMAGE_NAME"

echo "$TRAVIS_BRANCH"
echo "${TRAVIS_BRANCH}"

if [[ "${TRAVIS_BRANCH}" == "DEV" ]];
then
    _imageTag="DEV"
else
    source _version.py
    _imageTag=${__version__}
fi
echo "Image Tag: $_imageTag"

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
# Ask Docker to tag the image as latest and with the custom tag            #
############################################################################
echo "Tagging the image as $REGISTRY_REGION/$REGISTRY_NAMESPACE/$IMAGE_NAME:$IMAGE_TAG "
docker tag $IMAGE_NAME:$IMAGE_TAG $REGISTRY_REGION/$REGISTRY_NAMESPACE/$IMAGE_NAME:$IMAGE_TAG
if [ $TRAVIS_BRANCH == "main" ]; then
  echo "and $REGISTRY_REGION/$REGISTRY_NAMESPACE/$IMAGE_NAME:latest"
  docker tag $IMAGE_NAME:$IMAGE_TAG $REGISTRY_REGION/$REGISTRY_NAMESPACE/$IMAGE_NAME:latest
fi

############################################################################
# Push the image                                                           #
############################################################################
echo "Pushing image to registry"
docker push $REGISTRY_REGION/$REGISTRY_NAMESPACE/$IMAGE_NAME --all-tags

############################################################################
# end script                                                               #
############################################################################

