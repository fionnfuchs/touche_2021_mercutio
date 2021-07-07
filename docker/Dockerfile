FROM python:3.8-slim

RUN apt-get update && apt-get install -y build-essential

RUN mkdir /app
RUN mkdir /app/material
ADD src /app/src
ADD Pipfile* /app/
ADD docker/docker_config.yaml /app/config.yaml
ADD log.ini /app/log.ini
ADD material /app/material

RUN pip3 install pipenv
RUN cd /app && pipenv install --system --deploy

RUN python -m spacy download "en_core_web_lg"
RUN python /app/material/model_installer.py

ENV TRANSFORMERS_CACHE="/app/models/huggingface/transformers/"

WORKDIR /app

ENTRYPOINT ["python" ,"src/main.py"]
