FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

COPY . /gaussian-opacity-fields
WORKDIR /gaussian-opacity-fields
RUN apt update && apt install -y python python-dev python3-dev python3-pip 
RUN apt update && apt install -y python-is-python3
RUN pip install --no-cache-dir torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

#install cudatoolkit
RUN apt update && apt install -y wget
RUN pip install -r requirements.txt

ARG TORCH_CUDA_ARCH_LIST
RUN pip install submodules/diff-gaussian-rasterization
RUN pip install submodules/simple-knn/

# tetra-nerf for triangulation
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y cmake libgmp-dev libcgal-dev git 
ENV CPATH=/usr/local/cuda/targets/x86_64-linux/include:$CPATH
ENV LIBTORCH_PATH=/usr/local/lib/python3.8/dist-packages/torch \
    CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$LIBTORCH_PATH/lib:/usr/local/lib \
    Torch_DIR=$LIBTORCH_PATH \
    PATH=/usr/local/cuda/bin:$PATH \
    CUDA_BIN_PATH=/usr/local/cuda/bin

RUN cd submodules/tetra-triangulation && cmake .
RUN cd submodules/tetra-triangulation && /usr/bin/c++  -DBOOST_ALL_NO_LIB -Dtetranerf_cpp_extension_EXPORTS -I/include -I/usr/local/lib/python3.8/dist-packages/torch/include -I/usr/local/lib/python3.8/dist-packages/torch/include/torch/csrc/api/include -I/usr/local/lib/python3.8/dist-packages/torch/include/TH -I/usr/local/lib/python3.8/dist-packages/torch/include/THC -isystem /gaussian-opacity-fields/submodules/tetra-triangulation/_deps/pybind11-src/include -isystem /usr/include/python3.8  -Wno-deprecated-declarations -D_GLIBCXX_USE_CXX11_ABI=0 -O3 -DNDEBUG -fPIC -fvisibility=hidden   -flto -fno-fat-lto-objects -frounding-math -std=gnu++17 -o CMakeFiles/tetranerf_cpp_extension.dir/src/triangulation.cpp.o -c /gaussian-opacity-fields/submodules/tetra-triangulation/src/triangulation.cpp
RUN cd submodules/tetra-triangulation && /usr/bin/c++  -DBOOST_ALL_NO_LIB -Dtetranerf_cpp_extension_EXPORTS -I/include -I/usr/local/lib/python3.8/dist-packages/torch/include -I/usr/local/lib/python3.8/dist-packages/torch/include/torch/csrc/api/include -I/usr/local/lib/python3.8/dist-packages/torch/include/TH -I/usr/local/lib/python3.8/dist-packages/torch/include/THC -isystem /gaussian-opacity-fields/submodules/tetra-triangulation/_deps/pybind11-src/include -isystem /usr/include/python3.8  -Wno-deprecated-declarations -D_GLIBCXX_USE_CXX11_ABI=0 -O3 -DNDEBUG -fPIC -fvisibility=hidden   -flto -fno-fat-lto-objects -frounding-math -std=gnu++17 -o CMakeFiles/tetranerf_cpp_extension.dir/src/py_binding.cpp.o -c /gaussian-opacity-fields/submodules/tetra-triangulation/src/py_binding.cpp
RUN cd submodules/tetra-triangulation && /usr/bin/c++ -fPIC  -Wno-deprecated-declarations -D_GLIBCXX_USE_CXX11_ABI=0 -O3 -DNDEBUG -flto=auto -flto -shared  -o tetranerf/utils/extension/tetranerf_cpp_extension.cpython-38-x86_64-linux-gnu.so CMakeFiles/tetranerf_cpp_extension.dir/src/triangulation.cpp.o CMakeFiles/tetranerf_cpp_extension.dir/src/py_binding.cpp.o   -L/usr/local/lib/python3.8/dist-packages/torch/lib  -Wl,-rpath,/usr/local/lib/python3.8/dist-packages/torch/lib -lc10 -ltorch -ltorch_cpu -ltorch_python /usr/lib/x86_64-linux-gnu/libmpfr.so /usr/lib/x86_64-linux-gnu/libgmp.so 
RUN cd submodules/tetra-triangulation && /usr/bin/strip /gaussian-opacity-fields/submodules/tetra-triangulation/tetranerf/utils/extension/tetranerf_cpp_extension.cpython-38-x86_64-linux-gnu.so
RUN cd submodules/tetra-triangulation && pip install -e .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /workspace