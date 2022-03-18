# exatrkx-acat2021
The exa.trkx pipeline used for the C++ inference studies presented at ACAT 2021


# To compile the C++ pipeline

## Install dependencies
The C++ pipeline currently only runs on GPUs and depends on `cuda`, `libTorch`, `onnxrumtime`, and `cugraph`.

We prepared a docker container, `docexoty/exatrkx:tf2.5-torch1.9-cuda11.2-ubuntu20.04-rapids21.10-devel-hep`, 
which contains all dependences as well as those
needed by the [ACTS framework](https://github.com/acts-project/acts).

We also provide a [Dockerfile](docker/Dockerfile) for those who want to build one themselves.
```
docker build -t my-docker-hub-name/docker-name -f docker/Dockerfile
```

## Compile the code
First launch an interactive docker container

```bash
docker run -it --rm -v $PWD:$PWD -w $PWD --gpus all docexoty/exatrkx:tf2.5-torch1.9-cuda11.2-ubuntu20.04-rapids21.10-devel-hep bash
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
CMake Error at CMakeLists.txt:147 (target_link_libraries):                                                                                                           
  Error evaluating generator expression:                                                                                                                             
                                                                                                                                                                     
    $<TARGET_PROPERTY:torch_cpu,INTERFACE_COMPILE_DEFINITIONS>                    
                                                                                                                                                                     
  Target "torch_cpu" not found.  
```

The `cmake` command should ends with the following message:
```
-- Generating done                                                                                                                                                   
CMake Generate step failed.  Build files cannot be regenerated correctly.  
```

If so, go ahead and compile the code via `make -j4`.

If everything works so far, you can test the code via `./bin/inference`.
The following is the expected print out
```bash
Building and running a GPU inference engine for Embedding
314 spacepoints.
First spacepoint information: 0.0105378 -0.926339 -0.223784 
Embedding space of the first SP: -0.124342 0.0474498 -1.24684 0.43473 1.26207 0.451178 0.537944 -0.0946749 
Inserting points
Prefix Sum
Counting sorted
Neigbours to Edges
copy edges to std::vector
Built 2334 edges.
2 3 5 5 6 6 7 7 7 7 
1 2 2 3 1 0 1 2 3 5 
Get scores for 2334 edges.
After filtering: 584 edges.
Weakly components Start
edge size: 584 584
vertices size: 313
create graph from edgelist
After1: 
back from construct_graph
Number of Nodes:313
Number of Edges:584
Graph sym: 1
2back from construct_graph
number of components: 313
size of components: 314
22 reconstructed tracks.
```