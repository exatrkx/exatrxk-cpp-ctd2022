# ExaTrkX in C++

This repository houses the C++ implementation of the [ExaTrkX](https://arxiv.org/abs/2103.06995) pipeline
and the prototype implementation of the ExaTrkX as a service. 


The C++ implementation is different from our previous implementation [ACAT2021](https://github.com/exatrkx/exatrkx-acat2021) in
that previous implementation relies on [onnxruntime](https://github.com/microsoft/onnxruntime)
 and current one on [TorchScript](https://pytorch.org/tutorials/advanced/cpp_export.html).

The repository is organized as follows. `inf-cpp` contains the C++ implementation of the ExaTrkX pipeline and `triton_example` contains the Triton model configurations. 

# The ExaTrkX as a service
We use the [Triton server](https://github.com/triton-inference-server) to 
perform the ExaTrkX pipeline and to schedule requests from clients.
The triton server models can be found in the folder `triton_examples/models`.

First get `python_backend` environments from [here](), and put them into the folder `pymodels`.
Find the absolute path for `pymodels` and `triton_example/models` as required by Triton.
And replace `pymodels-abspath` and `triton_example/models-abspath` in the following command.

To launch the server:
```bash
docker run -it --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size '1gb' \
  -v pymodels-abspath:/pymodels \
  -v triton_example/models-abspath:/models \
    nvcr.io/nvidia/tritonserver:22.02-py3 tritonserver --model-repository=/models/
```

Triton will prinout the following in the terminal.
```bash
+-------------+---------+--------+
| Model       | Version | Status |
+-------------+---------+--------+
| applyfilter | 1       | READY  |
| embed       | 1       | READY  |
| exatrkx     | 1       | READY  |
| filter      | 1       | READY  |
| frnn        | 1       | READY  |
| gnn         | 1       | READY  |
| wcc         | 1       | READY  |
+-------------+---------+--------+
```
and 
```
I0531 20:45:26.837422 1 grpc_server.cc:4375] Started GRPCInferenceService at 0.0.0.0:8001
I0531 20:45:26.838380 1 http_server.cc:3075] Started HTTPService at 0.0.0.0:8000
I0531 20:45:26.885409 1 http_server.cc:178] Started Metrics Service at 0.0.0.0:8002
```