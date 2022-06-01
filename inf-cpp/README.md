# Introduction
The ExaTrkX pipeline used for the C++ inference.


# To compile the C++ pipeline

First launch an interactive docker container

```bash
docker run -it --rm --gpus all --net=host \
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

If everything works so far, you can test the code via `./bin/inference -h`.
The following is the expected print out
```bash
Usage: ./bin/inference [-hv] [-d input_file_path] [-s server_type]
 -s: server type. 0: no server, 1: torch, 2: python, 3: one, 4: combined
 -d: input data/directory
 -t: number of threads
 -v: verbose
```
To run a test locally: `./bin/inference -s 0`.