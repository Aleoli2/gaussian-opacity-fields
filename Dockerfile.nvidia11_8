FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

COPY . /gaussian-opacity-fields
WORKDIR /gaussian-opacity-fields
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y python3.9 python3.9-dev python3-dev python3-pip 
RUN apt update && apt install -y python-is-python3
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3 
RUN pip install --no-cache-dir torch==2.3.0+cu118 torchvision==0.18.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

#install cudatoolkit
RUN apt update && apt install -y wget
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ARG TORCH_CUDA_ARCH_LIST
RUN pip install submodules/diff-gaussian-rasterization
RUN pip install submodules/simple-knn/

# tetra-nerf for triangulation
RUN apt update && apt install -y cmake libgmp-dev libcgal-dev git 
ENV CPATH=/usr/local/cuda/targets/x86_64-linux/include:$CPATH

RUN cd submodules/tetra-triangulation && cmake . && make
RUN cd submodules/tetra-triangulation && pip install -e .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /workspace