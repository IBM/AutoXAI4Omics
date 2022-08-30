# Set base image and key env vars
FROM ubuntu:18.04
ENV DEBIAN_FRONTEND="noninteractive"
ARG USER_ID=${USER_ID}
ENV R_BASE_VERSION 4.2.0
ENV TF_CPP_MIN_LOG_LEVEL '2'

# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN apt-get install -y software-properties-common git

# Install python 3.7 & upgrade pip
RUN apt-get install -y python3.7 python3.7-dev python3.7-distutils python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN update-alternatives --set python /usr/bin/python3.7
RUN python -m pip install --upgrade pip

# Install R
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/'
RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN apt-get install -y fort77 r-base-core r-base r-base-dev r-recommended swig

# Add omicuser and set env vars
RUN useradd -l -m -s /bin/bash -u ${USER_ID} omicsuser
USER omicsuser
WORKDIR /home/omicsuser
ENV PATH "/home/omicsuser/.local/bin:${PATH}"
ENV PYTHONPATH "/home/omicsuser:${PYTHONPATH}"
ENV R_LIBS_USER=/home/omicsuser/.local/R
RUN mkdir -p ${R_LIBS_USER}

# Copy in required files
COPY --chown=omicsuser:omicsuser install_R_packages.sh .
COPY --chown=omicsuser:omicsuser *.py *.R ./
COPY --chown=omicsuser:omicsuser tabauto ./tabauto
COPY --chown=omicsuser:omicsuser omics ./omics
COPY --chown=omicsuser:omicsuser logging.yml ./

# Install required Python packages use block below if fixing other packages for the first time, use other
# COPY --chown=omicsuser:omicsuser requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=omicsuser:omicsuser requirements_fixed.txt .
RUN pip install -r requirements_fixed.txt  --no-cache-dir

# Install required R packages
RUN ./install_R_packages.sh

# use this a fix for Calour
RUN pip install numpy==1.20.0

# use root if dev work is needed
# user root

# init run command
CMD ["$@"]
