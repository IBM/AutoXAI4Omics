FROM python:3.7

RUN \
    apt update && \
    apt install -y r-base && \
    apt upgrade -y && \
    groupadd -r aotuser && useradd -m -s /bin/bash -g aotuser aotuser

USER aotuser

WORKDIR /home/aotuser

ENV PATH "/home/aotuser/.local/bin:${PATH}"

COPY --chown=aotuser:aotuser requirements.txt .

RUN \
     python -m pip install --upgrade pip && \
     cat requirements.txt | grep -v ^# | xargs -n 1 pip install

COPY --chown=aotuser:aotuser *.py *.R ./

ENV PYTHONPATH "/home/aotuser:${PYTHONPATH}"

CMD ["$@"]