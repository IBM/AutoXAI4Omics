# Copyright 2024 IBM Corp.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Set base image and key env vars
FROM python:3.9.14
# ENV DEBIAN_FRONTEND="noninteractive"

# Default 1001 - non privileged uid
ARG USER_ID=1001
ENV TF_CPP_MIN_LOG_LEVEL '2'
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION 'python'
# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN apt-get install -y software-properties-common git

# upgrade pip
RUN python -m pip install --upgrade pip

# Add omicuser and set env vars
# Give omicsuser gid 0 so has root group permissions to read files, 
#   and is the same gid as Openshift users. Compatible with Openshift and k8s
RUN useradd -l -m -s /bin/bash --uid ${USER_ID} -g 0 omicsuser

WORKDIR /home/omicsuser

# Install required Python packages use block below if fixing other packages for the first time, use other
COPY requirements.txt .
RUN pip install -r requirements.txt 

# grant write permissions to these folders
COPY --chown=omicsuser:0 src .

# Use 'omicsuser' user - this is overruled in Openshift
USER omicsuser
# init run command
CMD ["$@"]
