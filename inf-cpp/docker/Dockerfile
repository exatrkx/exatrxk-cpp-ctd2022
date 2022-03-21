# docexoty/exatrkx:triton-rapids
ARG TRITON_VERSION=22.02-py3-sdk
FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION}

ARG PYTORCH_VERSION=1.11.0 
ARG CUDANAME=cu115

# RUN apt-get update -y && apt-get -y install \
#     doxygen

RUN /usr/bin/python3 -m pip install --upgrade pip && pip install torch==${PYTORCH_VERSION}+${CUDANAME} \
    -f https://download.pytorch.org/whl/${CUDANAME}/torch_stable.html

RUN cd /tmp && \
    git clone --recursive https://github.com/pytorch/kineto.git && \
    cd kineto/libkineto &&  mkdir build && cd build && cmake .. \
        -DCMAKE_C_COMPILER=`which gcc` \
        -DCMAKE_CXX_COMPILER=`which g++` \
        -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j4 && make install && rm -rf /tmp/*

COPY deps.tar.gz /usr/local 
RUN cd /usr/local && tar xzf deps.tar.gz && rm deps.tar.gz
RUN apt-get update -y && apt-get install -y libboost-all-dev