FROM python:3.7

RUN \
    apt update && \
    apt install -y r-base && \
    apt upgrade -y

WORKDIR /aot_eai_omics

COPY requirements.txt .

RUN \
     python -m pip install --upgrade pip && \
     while read requirement; do pip install ${requirement}; done < requirements.txt

COPY *.py *.R ./

ENV PYTHONPATH "/aot_eai_omics:${PYTHONPATH}"

CMD ["$@"]