FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# https://serverfault.com/questions/683605/docker-container-time-timezone-will-not-reflect-changes
ENV TZ America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y git && pip install -U pip setuptools dvc dvc-gs

WORKDIR /app
COPY ./src /app
RUN pip install -e ./core
RUN pip install -e ./data
RUN pip install -e ./model
RUN pip install -e ./evaluation
