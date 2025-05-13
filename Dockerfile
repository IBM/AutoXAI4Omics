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
FROM python:3.11.12
RUN apt-get update && apt-get upgrade -y && apt-get clean

ENV TF_CPP_MIN_LOG_LEVEL='2'
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
ARG USER_ID=1001
RUN useradd -l -m -s /bin/bash --uid ${USER_ID} -g 0 omicsuser

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_HOME='/usr/local' \
    POETRY_VERSION=2.1.3

RUN curl -sSL https://install.python-poetry.org | python3 -
WORKDIR /home/omicsuser



COPY --chown=omicsuser:0 poetry.lock pyproject.toml ./
COPY --chown=omicsuser:0 autoxai4omics .

RUN poetry env use system
RUN poetry install --no-root --only main
USER omicsuser 

CMD ["$@"]