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

COPY tflite_runtime_nightly-2.14.0.dev20230508-cp310-cp310-manylinux2014_x86_64.whl /tmp/
RUN pip install --no-cache-dir /tmp/tflite_runtime_nightly-2.14.0.dev20230508-cp310-cp310-manylinux2014_x86_64.whl \
    && rm /tmp/tflite_runtime_nightly-2.14.0.dev20230508-cp310-cp310-manylinux2014_x86_64.whl

RUN git config --global --add safe.directory /workspace
