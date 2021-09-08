"""
    Class to build a TensorRT engine from ONNX weights.
"""

import argparse

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


def build_engine(config, calibrator=None):
    """ Build TensorRT engine from path and save to specified file
        NVIDIA guide:
            - https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#build_engine_python
        @param onnx_file_path:  Path to ONNX weight file
        @param trt_engine_path: Path to serialised TensorRT save location
    """
    # Logger to capture errors, warnings,
    # and other information during the build and inference phases
    TRT_LOGGER = trt.Logger()
    # initialize TensorRT engine and parse ONNX model
    network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(network_creation_flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # parse ONNX weights
    with open(config.onnx_file_path, "rb") as model:
        print("Beginning ONNX file parsing")
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
    print("Completed parsing of ONNX file")

    last_layer = network.get_layer(network.num_layers - 1)
    # Check if last layer recognizes it's output
    if not last_layer.get_output(0):
        # If not, then mark the output using TensorRT API
        network.mark_output(last_layer.get_output(0))

    # Allow TensorRT to use up to 1GB of GPU memory for tactic selection
    # Based on the NVIDIA guide, check the docstring above
    builder.max_workspace_size = 1 << 30
    # Online batch
    builder.max_batch_size = 1
    # Use FP16 mode if possible
    if config.mode == 'fp16':
        print('Using FP16')
        assert builder.platform_has_fast_fp16
        builder.fp16_mode = True

    # Generate TensorRT engine optimized for the target platform
    print("Building an engine... May take a few minutes...")
    engine = builder.build_cuda_engine(network)
    engine.create_execution_context()
    print("Completed creating Engine")

    # Save engine to file
    with open(config.trt_engine_path, "wb") as f:
        f.write(engine.serialize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a TensorRT engine from ONNX weights and save to file"
    )
    parser.add_argument("onnx_file_path", type=str, help="path to the onnx weight file")
    parser.add_argument(
        "trt_engine_path", type=str, help="path to save the trt engine file"
    )
    parser.add_argument('mode', default='fp32', type=str, help='fp16 (leave blank for fp32)')
    args = parser.parse_args()

    calibrator = None
    build_engine(args, calibrator)