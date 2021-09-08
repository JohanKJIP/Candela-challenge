"""
    Class to perform YOLO detections using TensorRT.
"""

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import rospy
import tensorrt as trt

from utils import *


class HostDeviceMem(object):
    """ Helper data class. """

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtDetector(object):
    COLOUR_CHANNELS = 3  # RGB

    def __init__(self, engine_path, classes, img_size, conf_thresh, nms_thresh):
        """ Init function
            @param engine_path:  Path to the TensorRT serialised engine
            @param classes:      Names of the classes
            @param img_size:     Size of each image dimension
            @param conf_thres:   Threshold on the detection confidence
            @param nms_thres:    Threshold for Non Maximum Suppression
        """
        self.TRT_LOGGER = trt.Logger()
        self.img_size = img_size
        self.classes = classes
        self.number_classes = len(self.classes)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self.engine = self.get_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.buffers = self.allocate_buffers(batch_size=1)

        self.context.set_binding_shape(0, (1, self.COLOUR_CHANNELS, img_size, img_size))
        rospy.loginfo("[cone_detection_camera] TRT initialisation complete")

    def get_engine(self, engine_path):
        """ Load serialised engine from file """
        rospy.loginfo("[cone_detection_camera] Reading engine from file {}".format(engine_path))
        with open(engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self, batch_size):
        """ Allocate necessary buffers for inference on the GPU
            @param batch_size: Size of the batches
            @return bounding_boxes
                - inputs: buffer for inputs
                - outputs: buffer for outputs
                - bindings: device bindings
                - stream: GPU stream, sequence of operations
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * batch_size

            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings
            bindings.append(int(device_mem))
            # Append to the appropriate list
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def do_inference(self, bindings, inputs, outputs, stream):
        """ Detect cones on the GPU """
        # Transfer input data to the GPU
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference
        self.context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs
        return [out.host for out in outputs]

    def predict(self, img):
        """ Detect classes in a given image
            @param img: Input image
            @return bounding_boxes
                - bounding_boxes: detected bounding boxes
        """
        # Pre-processing
        img_in = pre_process(img, self.img_size)

        # Do inference on GPU
        inputs, outputs, bindings, stream = self.buffers
        inputs[0].host = img_in

        trt_outputs = self.do_inference(
            bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        # Post process
        # (42588,) -> (1, 10647, 1, 4)
        #       - where the tuple is (batch, num, 1, 4)
        #       - where (1, 4) is a bounding box [x, y, w, h]
        trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
        # (42588,) -> (1, 10647, 4)
        #       - where the tuple is (batch, num, num_classes)
        trt_outputs[1] = trt_outputs[1].reshape(1, -1, self.number_classes)

        batch_bounding_boxes = post_processing(self.conf_thresh, self.nms_thresh, trt_outputs)

        # batch_size 1 thus index 0 to retrieve the only batch
        return batch_bounding_boxes[0]


if __name__ == "__main__":
    classes = load_classes('config/classes.names')
    detector = TrtDetector('yolov4.trt', classes, 608, 0.45, 0.45)
    img = cv2.imread("yolov4/test.jpg")
    bbs = detector.predict(img)
    bbs = rescale_bbs(img, bbs)
    img = plot_boxes_cv2(img, bbs, class_names=classes)
    cv2.imshow("boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
