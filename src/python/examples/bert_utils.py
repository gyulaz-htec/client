#!/usr/bin/env python

import argparse
import os
import sys
from functools import partial

import numpy as np
import time
import scipy.io
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import InferenceServerException, triton_to_np_dtype

def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for BERT (as expected by this client)
    """
    if len(model_metadata.inputs) != 3:
        raise Exception("expecting 3 inputs, got {}".format(len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 2:
        raise Exception(
            "expecting 2 outputs, got {}".format(len(model_metadata.outputs))
        )

    if len(model_config.input) != 3:
        raise Exception(
            "expecting 3 inputs in model configuration, got {}".format(
                len(model_config.input)
            )
        )

    input_names = []
    input_formats = []
    input_datatypes = []
    output_names = []

    for i in range(len(model_metadata.outputs)):
        output_metadata = model_metadata.outputs[i]

        if output_metadata.datatype != "FP32":
            raise Exception(
                "expecting output datatype to be FP32, model '"
                + model_metadata.name
                + "' output type is "
                + output_metadata.datatype
            )

        # Output is expected to be a vector. But allow any number of
        # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
        # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
        # is one.
        output_batch_dim = model_config.max_batch_size > 0
        non_one_cnt = 0
        for dim in output_metadata.shape:
            if output_batch_dim:
                output_batch_dim = False
            elif dim > 1:
                non_one_cnt += 1
                if non_one_cnt > 1:
                    raise Exception("expecting model output to be a vector")
        
        output_names.append(output_metadata.name)
        
    for i in range(len(model_metadata.inputs)):

        input_metadata = model_metadata.inputs[i]
        input_config = model_config.input[i]
    

        # Model input must have 3 dims, either CHW or HWC (not counting
        # the batch dimension), either CHW or HWC
        input_batch_dim = model_config.max_batch_size > 0
        expected_input_dims = 1 + (1 if input_batch_dim else 0)
        if len(input_metadata.shape) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".format(
                    expected_input_dims, model_metadata.name, len(input_metadata.shape)
                )
            )

        if input_metadata.datatype != "INT64":
            raise Exception(
                "expecting input datatype to be INT64, model '"
                + model_metadata.name
                + "' output type is "
                + input_metadata.datatype
            )

        input_names.append(input_metadata.name)
        input_formats.append(input_config.format)
        input_datatypes.append(input_metadata.datatype)

    return (
        model_config.max_batch_size,
        input_names,
        input_formats,
        input_datatypes,
        output_names,
    )

def requestGenerator(FLAGS, batched_data, input_names, output_names, dtypes):
    protocol = FLAGS.protocol.lower()

    if protocol == "grpc":
        client = grpcclient
    else:
        client = httpclient

    # Set the input data
    # print(f"## batched_data keys: {list(batched_data.keys())}")
    # print(f"## input_names: {input_names}")
    inputs = []
    for i in range (len(input_names)):
        # print(f"### {input_names[i]} shape: {batched_data[input_names[i]].shape} dytpe: {dtypes[i]}")
        inputs.append(client.InferInput(input_names[i], batched_data[input_names[i]].shape, dtypes[i]))
        inputs[i].set_data_from_numpy(batched_data[input_names[i]])
        # print(f"#### inputs[i]:\n{batched_data[input_names[i]]}")

    outputs = []
    # print(f"## output_names: {output_names}")
    for i in range (len(output_names)):
        outputs.append(client.InferRequestedOutput(output_names[i]))

    yield inputs, outputs


def convert_http_metadata_config(_metadata, _config):
    # NOTE: attrdict broken in python 3.10 and not maintained.
    # https://github.com/wallento/wavedrompy/issues/32#issuecomment-1306701776
    try:
        from attrdict import AttrDict
    except ImportError:
        # Monkey patch collections
        import collections
        import collections.abc

        for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))
        from attrdict import AttrDict

    return AttrDict(_metadata), AttrDict(_config)

class TritonBertClient:
    def __init__(self, FLAGS):
        if FLAGS.streaming and FLAGS.protocol.lower() != "grpc":
            raise Exception("Streaming is only allowed with gRPC protocol")

        try:
            if FLAGS.protocol.lower() == "grpc":
                # Create gRPC client for communicating with the server
                triton_client = grpcclient.InferenceServerClient(
                    url=FLAGS.url, verbose=FLAGS.verbose
                )
            else:
                # Specify large enough concurrency to handle the
                # the number of requests.
                concurrency = 20 if FLAGS.async_set else 1
                triton_client = httpclient.InferenceServerClient(
                    url=FLAGS.url, verbose=FLAGS.verbose, concurrency=concurrency
                )
        except Exception as e:
            print("client creation failed: " + str(e))
            sys.exit(1)

        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        try:
            model_metadata = triton_client.get_model_metadata(
                model_name=FLAGS.model_name, model_version=FLAGS.model_version
            )
        except InferenceServerException as e:
            print("failed to retrieve the metadata: " + str(e))
            sys.exit(1)

        try:
            model_config = triton_client.get_model_config(
                model_name=FLAGS.model_name, model_version=FLAGS.model_version
            )
        except InferenceServerException as e:
            print("failed to retrieve the config: " + str(e))
            sys.exit(1)

        if FLAGS.protocol.lower() == "grpc":
            model_config = model_config.config
        else:
            model_metadata, model_config = convert_http_metadata_config(
                model_metadata, model_config
            )

        max_batch_size, self.input_names, self.formats, self.dtypes, self.output_names = parse_model(
            model_metadata, model_config
        )

        self.supports_batching = max_batch_size > 0
        if not self.supports_batching and FLAGS.batch_size != 1:
            print("ERROR: This model doesn't support batching.")
            sys.exit(1)

        self.triton_client = triton_client
        self.model_metadata = model_metadata
        self.model_config = model_config
        self.responses = []
        self.sent_count = 0

    def infer(self, FLAGS, batched_data):
        try:
            for inputs, outputs in requestGenerator(
                FLAGS, batched_data, self.input_names, self.output_names, self.dtypes
            ):
                self.sent_count += 1
                return self.triton_client.infer(
                    FLAGS.model_name,
                    inputs,
                    request_id=str(self.sent_count),
                    model_version=FLAGS.model_version,
                    outputs=outputs,
                    timeout=6000000000,
                )
        except InferenceServerException as e:
            print("Inference failed: " + str(e))
            sys.exit(1)
