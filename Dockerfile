FROM python:3.7

ARG USER_ID=${USER_ID}

WORKDIR /root

RUN \
	echo "deb https://cloud.r-project.org/bin/linux/debian buster-cran40/" >> /etc/apt/sources.list && \
	apt-key adv --keyserver keyserver.ubuntu.com --recv-key E19F5F87128899B192B1A2C2AD5F960A256A04AF && \
	apt update && \
	apt upgrade -y && \
	apt install -y -t buster-cran40 fort77 r-base r-base-core r-recommended r-base-dev swig && \
	useradd -l -m -s /bin/bash -u ${USER_ID} omicsuser && \
	python -m pip install --upgrade pip

USER omicsuser

WORKDIR /home/omicsuser

ENV PATH "/home/omicsuser/.local/bin:${PATH}"

COPY --chown=omicsuser:omicsuser requirements.txt .

RUN \
	cat requirements.txt | \
	grep -v ^# | \
	xargs -n 1 pip install

COPY --chown=omicsuser:omicsuser install_Python_packages.sh .

RUN ./install_Python_packages.sh

ENV R_LIBS_USER=/home/omicsuser/.local/R

COPY --chown=omicsuser:omicsuser install_R_packages.sh .

RUN mkdir -p ${R_LIBS_USER}

USER root

RUN ./install_R_packages.sh

USER omicsuser

COPY --chown=omicsuser:omicsuser *.py *.R ./
ADD --chown=omicsuser:omicsuser tabauto ./tabauto

ENV PYTHONPATH "/home/omicsuser:${PYTHONPATH}"

CMD ["$@"]
