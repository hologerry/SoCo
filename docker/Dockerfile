# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------


FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y vim curl wget ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev \
    && apt-get install -y libblacs-mpi-dev \
    && apt-get install software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install python3.7 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python 3.7 to python and install pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 10
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py

# Torch
RUN pip install torch==1.6.0 torchvision==0.7.0

# accimage
RUN pip install --prefix=/opt/intel/ipp ipp-devel
RUN pip install git+https://github.com/pytorch/accimage

# apex
RUN git clone https://github.com/NVIDIA/apex 
RUN cd apex/
# P100, P40, V100 and 2080Ti
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5"
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# detectron2
RUN python -m pip install 'git+https://github.com/hologerry/detectron2.git' --upgrade --force-reinstall

# Install MMCV and MMdetection
RUN pip uninstall pycocotools
RUN pip install mmcv-full==1.2.4 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html --upgrade --force-reinstall

RUN git clone https://github.com/hologerry/mmdetection.git
RUN cd mmdetection
RUN pip install -r requirements/build.txt --upgrade --force-reinstall
RUN pip install -v -e . --upgrade --force-reinstall
