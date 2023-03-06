# Set base image and key env vars
FROM python:3.9.14
# ENV DEBIAN_FRONTEND="noninteractive"

# Default 1001 - non privileged uid
ARG USER_ID=1001
ENV TF_CPP_MIN_LOG_LEVEL '2'

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

# Copy in required files
COPY *.py ./
# grant write permissions to these folders
COPY --chown=omicsuser:0 tabauto ./tabauto 
COPY --chown=omicsuser:0 omics ./omics

COPY logging.yml ./

# Install required Python packages use block below if fixing other packages for the first time, use other
COPY requirements.txt .
RUN pip install -r requirements.txt 
# COPY --chown=omicsuser:omicsuser requirements_fixed.txt .
# RUN pip install -r requirements_fixed.txt 

# use this a fix for Calour
# RUN pip install numpy==1.20.0

# Use 'omicsuser' user - this is overruled in Openshift
USER omicsuser
# init run command
CMD ["$@"]
