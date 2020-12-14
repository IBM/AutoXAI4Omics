FROM python:3.7

ARG USER_ID=${USER_ID}

WORKDIR /root

RUN \
	echo "deb https://cloud.r-project.org/bin/linux/debian buster-cran40/" >> /etc/apt/sources.list && \
	apt-key adv --keyserver keys.gnupg.net --recv-key E19F5F87128899B192B1A2C2AD5F960A256A04AF && \
	apt update && \
	apt upgrade -y && \
	apt install -y -t buster-cran40 r-base r-base-core r-recommended r-base-dev swig && \
	useradd -l -m -s /bin/bash -u ${USER_ID} aotuser && \
	python -m pip install --upgrade pip

USER aotuser

WORKDIR /home/aotuser

ENV PATH "/home/aotuser/.local/bin:${PATH}"

COPY --chown=aotuser:aotuser requirements.txt .

RUN \
	cat requirements.txt | \
	grep -v ^# | \
	xargs -n 1 pip install

COPY --chown=aotuser:aotuser install_Python_packages.sh .

RUN ./install_Python_packages.sh

ENV R_LIBS_USER=/home/aotuser/.local/R

COPY --chown=aotuser:aotuser install_R_packages.sh .

RUN mkdir -p ${R_LIBS_USER}

USER root

RUN ./install_R_packages.sh

USER aotuser

COPY --chown=aotuser:aotuser *.py *.R ./
ADD --chown=aotuser:aotuser tabauto ./tabauto

ENV PYTHONPATH "/home/aotuser:${PYTHONPATH}"

CMD ["$@"]
