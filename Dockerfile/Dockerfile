FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace
RUN apt-get update \
    && apt-get install -y tzdata build-essential wget git git-lfs python3-pip libgl1-mesa-dev libglib2.0-0 \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN wget https://github.com/PINTO0309/onnx2tf/releases/download/1.7.3/flatc.tar.gz
RUN tar -zxvf flatc.tar.gz \
    && chmod +x flatc \
    && mv flatc /usr/bin/

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

RUN git config --global --add safe.directory /workspace
