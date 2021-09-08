"""
    Detect and count boats in video files.
"""

import os
import argparse
import time

import cv2

from sort.sort import *
from tracking import BoundingBoxTracker
from trt_detector import TrtDetector
from utils import *


class BoatDetector:

    def __init__(self):
        self.classes = load_classes('config/classes.names')
        self.img_size = 608
        self.conf_thres = 0.3
        self.nms_thres = 0.45
        self.trt_detector = TrtDetector('yolov4.trt', self.classes,
                                        self.img_size, self.conf_thres,
                                        self.nms_thres)
        self.tracker = Sort()

        self.colours = []
        for _ in range(32):
            h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
            self.colours.append([int(256*i) for i in colorsys.hls_to_rgb(h, l, s)])

    def remove_self_detections(self, bbs, img):
        """ Remove self detections on the boat
        @param bbs:         Bounding boxes to check
        @param img:         Image to overlay debug
        @return new_bbs, img
            - new_bbs:      Filtered bbs
            - img:          Image with exclusion region
        """
        x1_boat = 480
        x2_boat = 760
        y1_boat = 420
        y2_boat = 720

        new_bbs = []
        for bb in bbs:
            x_centre = (bb[0] + bb[2]) / 2
            y_centre = (bb[1] + bb[3]) / 2

            if not (x_centre > x1_boat and x_centre < x2_boat
                    and y_centre > y1_boat and y_centre < y2_boat):
                new_bbs.append(bb)

        # Overlay exlusion region
        sub_img = img[x1_boat:x2_boat, y1_boat:y2_boat]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
        img[x1_boat:x2_boat, y1_boat:y2_boat] = res

        return new_bbs, img

    def detect(self, options):
        """ Detect boats in a video
            @param file: Path to video
        """
        file = opt.video
        cap = cv2.VideoCapture(file)
        if not(cap.isOpened()):
            print("Error opening video stream or file: {0}".format(file))

        width = int(cap.get(3))
        height = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        file_name = file.split('.')[0].split('/')[1]
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter('output/{0}.avi'.format(file_name),
                              fourcc, fps, (width, height))

        while cap.isOpened():
            ret, img = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            start = time.time()
            bbs = self.trt_detector.predict(img)
            bbs = rescale_bbs(img, bbs)

            # Remove self detections on the boat
            if opt.remove_self:
                bbs, img = self.remove_self_detections(bbs, img)

            # Reshape to tracker input, only keep boat bbs (cls_id=8)
            bbs_xy = np.asarray([bb[0:5] for bb in bbs if bb[6] == 8])
            if len(bbs_xy) == 0:
                bbs_xy = np.empty((0, 5))
            trackers = self.tracker.update(bbs_xy)

            img = plot_boxes_cv2(img, trackers, bbs, self.colours, class_names=self.classes)
            img = plot_fps(img, time.time() - start)

            cv2.imshow('Video', img)
            out.write(img)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="videos/test_video.avi", help="video file name")
    parser.add_argument("--no-filter", dest='remove_self',  action='store_false', help="If remove self detections with exclusion region")
    opt = parser.parse_args()
    print(opt)
    boats = BoatDetector()
    boats.detect(opt)
