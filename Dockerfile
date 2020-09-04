FROM python:3.7

ARG USER_ID=${USER_ID}
ARG GROUP_ID=${GROUP_ID}

RUN \
    apt update && \
    apt install -y r-base && \
    apt upgrade -y && \
    groupadd -g ${GROUP_ID} aotuser && \
    useradd -l -m -s /bin/bash -u ${USER_ID} -g ${GROUP_ID} aotuser && \
    python -m pip install --upgrade pip

USER aotuser

WORKDIR /home/aotuser

ENV PATH "/home/aotuser/.local/bin:${PATH}"

COPY --chown=aotuser:aotuser requirements.txt .

RUN \
     cat requirements.txt | grep -v ^# | xargs -n 1 pip install

COPY --chown=aotuser:aotuser *.py *.R ./

ENV PYTHONPATH "/home/aotuser:${PYTHONPATH}"

CMD ["$@"]
