ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION=3.10
ARG YOLOX_REPO=https://github.com/Megvii-BaseDetection/YOLOX.git
ARG YOLOX_REF=main
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    git-lfs \
    libffi-dev \
    libjpeg-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    protobuf-compiler \
    pkg-config \
    sudo \
    zlib1g-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url ${TORCH_INDEX_URL}

WORKDIR /workspace
RUN git clone --depth 1 --branch ${YOLOX_REF} ${YOLOX_REPO} yolox

WORKDIR /workspace/yolox
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install --no-cache-dir cython pycocotools ninja
RUN python3 -m pip install --no-cache-dir --no-build-isolation mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.1/index.html
RUN python3 -m pip install --no-cache-dir --no-build-isolation .

ENV PYTHONUNBUFFERED=1
ENV YOLOX_HOME=/workspace/yolox

CMD ["bash"]
