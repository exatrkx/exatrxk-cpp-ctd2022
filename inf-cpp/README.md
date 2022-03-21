# exatrkx-acat2021
The exa.trkx pipeline used for the C++ inference


# To compile the C++ pipeline

First launch an interactive docker container

```bash
docker run -it --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $PWD:$PWD -w $PWD docexoty/exatrkx:triton-rapids bash
```

then 
```bash
mkdir build && cd build
```

then 
```bash
../make.sh
```

You will see many errors and warnings concerning `torch` such as follows, but you can safely ignore
them.
```
CMake Warning at CMakeLists.txt:83 (add_executable):                                                                                                                 
  Cannot generate a safe runtime search path for target inference because                                                                                            
  there is a cycle in the constraint graph:                                                                                                                          
                                                                                                                                                                     
    dir 0 is [/usr/local/cuda/lib64]                                                                                                                                 
      dir 2 must precede it due to runtime library [libnvToolsExt.so.1]                                                                                              
    dir 1 is [/home/xju/code/exatrkx-cpp/inf-cpp/build/lib]                                                                                                          
    dir 2 is [/opt/conda/envs/rapids/lib]                                                                                                                            
      dir 0 must precede it due to runtime library [libnvrtc.so.11.2]                                                                                                
    dir 3 is [/opt/conda/envs/rapids/lib/python3.9/site-packages/torch/lib]                                                                                          
      dir 0 must precede it due to runtime library [libcublas.so.11]                                                                                                 
      dir 2 must precede it due to runtime library [libcudnn.so.8]                
                                                                                  
  Some of these libraries may not be found correctly
```

If so, go ahead and compile the code via `make -j4`.

If everything works so far, you can test the code via `./bin/inference`.
The following is the expected print out
```bash
Building and running a GPU inference engine for Embedding
314 spacepoints.
First spacepoint information: 0.0105378 -0.926339 -0.223784 
Embedding space of libtorch the first SP: 
-0.1243  0.0474 -1.2468  0.4347  1.2621  0.4512  0.5379 -0.0947
[ CUDAFloatType{1,8} ]

Inserting points
Prefix Sum
Counting sorted
Neigbours to Edges
Built 2334 edges. 2
 2  3  5  5  6
 1  2  2  3  1
[ CPULongType{2,5} ]
Get scores for 2334 edges.
After filtering: 2334 1
 0.0869
 0.4881
 0.4880
 0.9469
 0.1455
 0.1091
 0.7052
 0.1135
 0.0882
[ CUDAFloatType{9} ]
After filtering: 584 edges.
 3  5  5  7  9
 2  2  3  1  1
[ CPULongType{2,5} ]
GNN scores for 584 edges.
 0.2066
 0.2166
 0.9292
 0.7839
 0.0437
[ CPUFloatType{5} ]
size of components: 314
22 reconstructed tracks.

```