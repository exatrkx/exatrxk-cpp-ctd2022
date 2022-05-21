from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import sys

import numpy as np
from numpy import loadtxt
model_name = "exatrkx"


with grpcclient.InferenceServerClient("localhost:8001") as client:
    #input0_data = np.random.rand(*shape).astype(np.float32)
    #input1_data = np.random.rand(*shape).astype(np.float32)

    file_o = open('data/in_e.csv','rb')
    input0_data = loadtxt(file_o,delimiter=",").astype(np.single)

    print(input0_data.dtype)

    inputs = [
        grpcclient.InferInput("FEATURES", input0_data.shape,
                              np_to_triton_dtype(input0_data.dtype))
    ]

    inputs[0].set_data_from_numpy(input0_data)

    outputs = [
        grpcclient.InferRequestedOutput("LABELS"),
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    result = response.get_response()
    output0_data = response.as_numpy("LABELS")
    print(output0_data.shape, output0_data.dtype)

    sys.exit(0)
