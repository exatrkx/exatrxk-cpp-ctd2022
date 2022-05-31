# ExaTrkX as a Service

This repository houses the C++ implementation of the [ExaTrkX](https://arxiv.org/abs/2103.06995) pipeline
and the prototype implementation of the ExaTrkX as a service. 


The C++ implementation is different from our previous implementation [ACAT2021](https://github.com/exatrkx/exatrkx-acat2021) in
that previous implementation relies on [onnxruntime](https://github.com/microsoft/onnxruntime)
 and current one on [TorchScript](https://pytorch.org/tutorials/advanced/cpp_export.html).

# The ExaTrkX as a service
We use the [Triton server](https://github.com/triton-inference-server) to 
perform the ExaTrkX pipeline and to schedule requests from clients.
The triton server models can be found in the folder `triton_examples/models`.

First get `python_backend` environments from here, and put them into the folder `pymodels`.
To launch the server:
```bash
docker run -it --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size '1gb' \
  -v triton_example/models:/models \
    nvcr.io/nvidia/tritonserver:22.02-py3 tritonserver --model-repository=/models/
```