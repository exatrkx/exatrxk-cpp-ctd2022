ARG RAPIDS_VERSION=21.10-cuda11.2-devel-ubuntu20.04-py3.8
FROM rapidsai/rapidsai-dev:${RAPIDS_VERSION}

ARG CMAKE_VERSION=3.21.1
ARG TF_VERSION=2.5

RUN source activate rapids && conda install -c conda-forge mamba \
    && mamba install -y swig cudnn=8.2.1 xtensor pybind11 

# Install cmake
RUN cd /tmp && source activate rapids \
    && wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh \
    && bash cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/opt/conda/envs/rapids --exclude-subdir --skip-license \
	&& rm -rf /tmp/*

RUN source activate rapids \
    && pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html \
    && pip install torch-sparse  -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html \
    && pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html \
    && pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html \
    && pip install torch-geometric \
    && pip install tensorflow_gpu==${TF_VERSION} tensorboard==2.6.0

# NV-system for profiling
# https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html#cuda-support
RUN apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         apt-transport-https \
         ca-certificates \
         gnupg \
         wget && \
     rm -rf /var/lib/apt/lists/*
RUN wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/nvidia.pub | apt-key add - && \
     echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/ /" >> /etc/apt/sources.list.d/nsight.list && \
     apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         nsight-systems-2021.2  && \
    rm -rf /var/lib/apt/lists/*

# nsight-compute-2021.1.0
# https://developer.nvidia.com/nsight-compute-history
# https://developer.nvidia.com/blog/using-nsight-compute-in-containers/
RUN  echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
     wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
         apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         nsight-compute-2021.1 && \
     rm -rf /var/lib/apt/lists/*

# install other inputs for exatrkx
RUN source activate rapids \
    && pip install pytorch-lightning \
    git+https://github.com/LAL/trackml-library.git \
    git+https://github.com/deepmind/graph_nets.git \
    matplotlib sklearn pyyaml>=5.1 tqdm networkx 

RUN source activate rapids \
    && cd /rapids && git clone https://github.com/xju2/prefix_sum.git \
    && git clone https://github.com/murnanedaniel/FRNN.git && cd prefix_sum \
    && NVCC_FLAGS="-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80" python setup.py install \
    && pip install . && cd /rapids/FRNN \
    && NVCC_FLAGS="-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80" python setup.py install \
    && pip install . 


RUN source activate rapids \
    && pip install wandb tables \
    && cd /rapids && git clone https://github.com/facebookresearch/faiss.git \
    && cd faiss && mkdir build && cd build \
    && cmake .. -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON \
        -DCMAKE_CUDA_ARCHITECTURES="60-real;70-real;75-real;80" \
        -DPython_EXECUTABLE=${CONDA_PREFIX}/bin/python \
        -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    && make -j faiss  && make -j swigfaiss \
    && cd faiss/python && pip install . \
    && cd /rapids/faiss/build && make install

ENV export LD_PRELOAD=/opt/conda/envs/rapids/lib/libmkl_def.so:/opt/conda/envs/rapids/lib/libmkl_avx2.so:/opt/conda/envs/rapids/lib/libmkl_core.so:/opt/conda/envs/rapids/lib/libmkl_intel_lp64.so:/opt/conda/envs/rapids/lib/libmkl_intel_thread.so:/opt/conda/envs/rapids/lib/libiomp5.so

# RUN rm /usr/lib/gcc/x86_64-linux-gnu/9/libstdc++.so \
#     && ln -s /opt/conda/envs/rapids/lib/libstdc++.so.6.0.28 /usr/lib/gcc/x86_64-linux-gnu/9/libstdc++.so

# the cudnn-home has to be /usr/lib/x86_64-linux-gnu/
# if use /opt/conda/envs/rapids, onnxruntime will use an incompatible gtest that is used in rapidsai
# copy all cudnn headers and libraries to /usr/lib/x86_64-linux-gnu/ seems solve the problem.
RUN cd /rapids && \
    git clone --recursive --branch v1.9.1 https://github.com/Microsoft/onnxruntime \
    && cd onnxruntime \
    && source activate rapids \
    && for FILE in /opt/conda/envs/rapids/include/cudnn*; \
        do ln -s $FILE /usr/include/$(basename $FILE); done \
    && for FILE in /opt/conda/envs/rapids/lib/libcudnn*.so; \
        do ln -s $FILE /usr/lib/x86_64-linux-gnu/$(basename $FILE); done \
    && ./build.sh --skip_submodule_sync \
        --cuda_home /opt/conda/envs/rapids \
        --cudnn_home /usr/lib/x86_64-linux-gnu/ \
        --use_cuda \
        --config Release \
        --build_wheel \
		--build_shared_lib \
		--skip_tests --skip_onnx_tests \
        --path_to_protoc_exe /opt/conda/envs/rapids/bin/protoc \
		--cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=60;70;75;80' \
            'CMAKE_INSTALL_PREFIX=/opt/conda/envs/rapids' \
            'BUILD_ONNX_PYTHON=ON' \
            'CMAKE_C_COMPILER=/usr/local/bin/gcc' \
            'CMAKE_CXX_COMPILER=/usr/local/bin/g++' \
        --parallel 3 --update --build && cd build/Linux/Release && \
		make install && pip install dist/*

RUN source activate rapids && mamba install onnx=1.10.0

RUN apt-get update -y && source activate rapids && apt-get -y install \
	dpkg-dev binutils libxpm-dev \
	libxft-dev libssl-dev \
	gfortran libpcre3-dev \
	xlibmesa-glu-dev libglew-dev libftgl-dev \
	libmysqlclient-dev libfftw3-dev \
	graphviz-dev \
	libldap2-dev libxml2-dev libkrb5-dev \
	libgsl0-dev libtbb2 libtbb-dev \
	&& rm -rf /var/lib/apt/lists/*

RUN cd /rapids && source activate rapids \
  	&& wget https://root.cern/download/root_v6.20.08.source.tar.gz -O- | tar xz \
  	&& mkdir root_builder && cd root_builder \
	&& cmake ../root-6.20.08 -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
		-Dminuit2=ON -Dgviz=ON -Drpath=ON -Dgnuinstall=ON -Dcxx17=ON \
		-DCMAKE_C_COMPILER=/usr/local/bin/gcc -DCMAKE_CXX_COMPILER=/usr/local/bin/g++ \
	&& cmake --build . --target install -- -j4

RUN cd /tmp && source activate rapids \
    && wget https://lhapdf.hepforge.org/downloads/?f=LHAPDF-6.2.3.tar.gz -O- | tar xz \
    && cd LHAPDF-6.2.3/ && CC=/usr/local/bin/gcc CXX=/usr/local/bin/g++ ./configure --prefix=${CONDA_PREFIX} \
    && make -j4 && make install && cd / && rm -r /tmp/*

# HepMC2-2.06.11
# HepMC3-3.2.1
RUN cd /tmp && source activate rapids \
    && wget http://hepmc.web.cern.ch/hepmc/releases/hepmc2.06.11.tgz -O- | tar xz \
    && cd HepMC-2.06.11 && mkdir build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
		-DCMAKE_C_COMPILER=/usr/local/bin/gcc \
		-DCMAKE_CXX_COMPILER=/usr/local/bin/g++ \
        -Dmomentum="GEV" -Dlength="MM" -Dbuild_docs=OFF\
	&& make -j && make install \
    && cd /tmp && wget http://hepmc.web.cern.ch/hepmc/releases/HepMC3-3.2.1.tar.gz -O- | tar xz \
    && cd /tmp/HepMC3-* && mkdir build && cd build  \
    && cmake .. -DHEPMC3_ENABLE_ROOTIO=OFF \
        -DHEPMC3_ENABLE_PYTHON=OFF \	
        -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
        -DHEPMC3_BUILD_EXAMPLES=OFF \
		-DCMAKE_C_COMPILER=/usr/local/bin/gcc \
		-DCMAKE_CXX_COMPILER=/usr/local/bin/g++ \
	&& make -j && make install \
    && rm -rf /tmp/*


# clhep
RUN cd /tmp && source activate rapids \
	&& wget http://proj-clhep.web.cern.ch/proj-clhep/dist1/clhep-2.4.1.3.tgz -O- | tar xz  \
	&& cd 2.4.1.3/CLHEP && mkdir build && cd build \
	&& cmake .. \
        -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
		-DCMAKE_C_COMPILER=/usr/local/bin/gcc \
		-DCMAKE_CXX_COMPILER=/usr/local/bin/g++ \
	&& make -j4 && make install \
	&& rm -rf /tmp/*

# geant 4
RUN cd /tmp && source activate rapids \
	&& wget http://geant4-data.web.cern.ch/geant4-data/releases/geant4.10.06.p02.tar.gz -O- | tar xz \
	&& cd geant4.10.06.p02 && mkdir build && cd build \
	&& cmake .. -DGEANT4_INSTALL_DATA=ON -DGEANT4_USE_GDML=ON \
				-DXERCESC_ROOT_DIR=${CONDA_PREFIX} -DCLHEP_ROOT_DIR=${CONDA_PREFIX} \
                -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
				-DCMAKE_C_COMPILER=/usr/local/bin/gcc \
				-DCMAKE_CXX_COMPILER=/usr/local/bin/g++ \
	&& make -j4 && make install \
    && rm -rf /tmp/*

ENV G4DATA=/opt/conda/envs/rapids/share/Geant4-10.6.2/data \
    G4ENSDFSTATEDATA=${G4DATA}/G4ENSDFSTATE2.2 \
    G4LEVELGAMMADATA=${G4DATA}/PhotonEvaporation5.5 \
    G4PARTICLEXSDATA=${G4DATA}/G4PARTICLEXS2.1 \
    G4RADIOACTIVEDATA=${G4DATA}/RadioactiveDecay5.4 \
    G4LEDATA=${G4DATA}/G4EMLOW7.9.1 \
    G4NEUTRONHPDATA=${G4DATA}/G4NDL4.6

# dd4hep 1.11.02 does not work
# https://github.com/AIDASoft/DD4hep/archive/v01-11-02.tar.gz
# 01-16-01 requires python exactly 3.8.12, hack the cmake
# https://github.com/AIDASoft/DD4hep/archive/refs/tags/v01-16-01.tar.gz

# TBB is completely messed up. 
# rapidsai contains one, ROOT installed one, and I just install another one above
# the code use find_packages(TBB required config), which did not find proper tbb config.
RUN cd /tmp && source activate rapids \
 	&& wget https://github.com/AIDASoft/DD4hep/archive/refs/tags/v01-16-01.tar.gz -O- | tar xz \
	&& cd DD4hep-* && sed -i 's/EXACT//g' cmake/DD4hepBuild.cmake  && mkdir build && cd build \ 
    && cmake .. -DDD4HEP_USE_XERCESC=ON -DDXERCESC_ROOT_DIR=${CONDA_PREFIX} \
			-DDD4HEP_USE_GEANT4=ON -DGeant4_DIR=${CONDA_PREFIX} \
			-DBUILD_TESTING=OFF \
			-DROOT_DIR=/opt/conda/envs/rapids \
			-DDD4HEP_USE_TBB=ON -DDD4HEP_USE_HEPMC3=ON \
           -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
			-DCMAKE_C_COMPILER=/usr/local/bin/gcc \
			-DCMAKE_CXX_COMPILER=/usr/local/bin/g++ \
	&& make -j4 && make install \
    && rm -rf /tmp/*

# Fastjet 3.4.0
RUN cd /tmp && source activate rapids \
    && wget http://fastjet.fr/repo/fastjet-3.4.0.tar.gz -O- | tar xz \
    && cd fastjet-3.4.0 && CC=/usr/local/bin/gcc CXX=/usr/local/bin/g++ ./configure --prefix=${CONDA_PREFIX} \
    && make -j4 && make install \
    && rm -rf /tmp/*

RUN apt-get install -y rsync

# Pythia 8.302
RUN cd /tmp && source activate rapids \
    && wget https://pythia.org/download/pythia83/pythia8302.tgz -O- | tar xz \
    && cd pythia*/ \
    && ./configure --with-fastjet3=${CONDA_PREFIX} \
           --with-hepmc2=${CONDA_PREFIX} --with-lhapdf6=${CONDA_PREFIX} \
           --with-root-bin=${CONDA_PREFIX}/root/bin --with-root-lib=${CONDA_PREFIX}/root/lib/root \
           --with-root-include=${CONDA_PREFIX}/root/include/root \
           --with-gzip-include=/usr/include --with-gzip-lib=/usr/lib64 \
           --with-hepmc3-include=${CONDA_PREFIX}/include \
           --with-hepmc3-lib=${CONDA_PREFIX}/lib64 --cxx=/usr/local/bin/g++ \
           --prefix=${CONDA_PREFIX} \
    && make -j4 && make install \
    && rm -rf /tmp/*

ENV PYTHIA8DATA=/opt/conda/envs/rapids/share/Pythia8/xmldoc/

# Dependencies for HPX
RUN apt-get update -y && apt-get install -y hwloc libasio-dev libtcmalloc-minimal4 libgoogle-perftools-dev


# for ACTS
RUN apt-get -y install libeigen3-dev gdb
# for pytorch
RUN source activate rapids && mamba install -y mkl-include