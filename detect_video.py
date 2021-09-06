from trt_detector import TrtDetector
from tracking import BoundingBoxTracker
from utils import *

import os
import argparse
import time

import cv2

class BoatDetector:

    def __init__(self):
        self.classes = load_classes('config/classes.names')
        self.img_size = 608
        self.conf_thres = 0.3
        self.nms_thres = 0.45
        self.trt_detector = TrtDetector('yolov4.trt', self.classes, 
                                    self.img_size, self.conf_thres, 
                                    self.nms_thres)
        self.tracker = BoundingBoxTracker()

    def detect(self, file):
        """ Detect boats in video
            @param file: Path to video
        """
        cap = cv2.VideoCapture(file)
        if (cap.isOpened() == False):
            print("Error opening video stream or file: {0}".format(file))

        width = int(cap.get(3))
        height = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        file_name = file.split('.')[0]
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter('output/{0}.avi'.format(file_name), 
                              fourcc, fps, (width, height))

        while cap.isOpened():
            ret, img = cap.read()
            if ret == True:
                start = time.time()
                bbs = self.trt_detector.predict(img)
                self.tracker.track(bbs)
                bbs = rescale_bbs(img, bbs)

                img = plot_boxes_cv2(img, bbs, self.tracker.color_list, class_names=self.classes)
                cv2.imshow('Video', img)
                print("{0}ms".format((time.time() - start)*1000))

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="test_video.avi", help="video file name")
    opt = parser.parse_args()
    boats = BoatDetector()
    boats.detect(opt.video)


