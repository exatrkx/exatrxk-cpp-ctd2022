# Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from tritonclient.utils import *
import tritonclient.http as httpclient
import sys

import numpy as np
from numpy import loadtxt
model_name = "wcc"
#shape = [4]

with httpclient.InferenceServerClient("localhost:8000") as client:
    file_i = open('data/out_fil_edge.csv','rb')
    input0_data = loadtxt(file_i,delimiter=",").astype(np.int64)
    print(input0_data.shape)
    file_o = open('data/out_gnn.csv','rb')
    input1_data = loadtxt(file_o,delimiter=" ").astype(np.float32)
    print(input1_data.shape)
    inputs = [
        httpclient.InferInput("NUMNODES", [1],
                              np_to_triton_dtype(np.int64)),
        httpclient.InferInput("INPUT0", input0_data.shape,
                              np_to_triton_dtype(input0_data.dtype)),
        httpclient.InferInput("INPUT1", input1_data.shape,
                              np_to_triton_dtype(input1_data.dtype)),
    ]

    inputs[0].set_data_from_numpy(np.array([314], dtype=np.int64))
    inputs[1].set_data_from_numpy(input0_data)
    inputs[2].set_data_from_numpy(input1_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    result = response.get_response()
    output0_data = response.as_numpy("OUTPUT0")

    #print("INPUT0 ({}) + INPUT1 ({}) = OUTPUT0 ({})".format(
    #    input0_data, input1_data, output0_data))
    #print("INPUT0 ({}) - INPUT1 ({}) = OUTPUT0 ({})".format(
    #    input0_data, input1_data, output1_data))

    #if not np.allclose(input0_data + input1_data, output0_data):
    #    print("add_sub example error: incorrect sum")
    #    sys.exit(1)

    #if not np.allclose(input0_data - input1_data, output1_data):
    #    print("add_sub example error: incorrect difference")
    #    sys.exit(1)

    #print('PASS: add_sub')
    print(output0_data.shape)
    sys.exit(0)
