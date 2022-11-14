# Set base image and key env vars
FROM python:3.9.14
# ENV DEBIAN_FRONTEND="noninteractive"
ARG USER_ID=${USER_ID}
ENV TF_CPP_MIN_LOG_LEVEL '2'

# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN apt-get install -y software-properties-common git

# upgrade pip
RUN python -m pip install --upgrade pip

# Add omicuser and set env vars
RUN useradd -l -m -s /bin/bash -u ${USER_ID} omicsuser
USER omicsuser
WORKDIR /home/omicsuser
ENV PATH "/home/omicsuser/.local/bin:${PATH}"
ENV PYTHONPATH "/home/omicsuser:${PYTHONPATH}"

# Copy in required files
COPY --chown=omicsuser:omicsuser *.py ./
COPY --chown=omicsuser:omicsuser tabauto ./tabauto
COPY --chown=omicsuser:omicsuser omics ./omics
COPY --chown=omicsuser:omicsuser logging.yml ./

# Install required Python packages use block below if fixing other packages for the first time, use other
COPY --chown=omicsuser:omicsuser requirements.txt .
RUN pip install -r requirements.txt 
# COPY --chown=omicsuser:omicsuser requirements_fixed.txt .
# RUN pip install -r requirements_fixed.txt 

# use this a fix for Calour
# RUN pip install numpy==1.20.0

# use root if dev work is needed
# user root

# init run command
CMD ["$@"]
