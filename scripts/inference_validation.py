"""
    Write inference results to file.
"""

import os
import shutil

import cv2

from trt_detector import TrtDetector
from utils import *


def run_inference(model, img_path, out_path, show_boxes=False):
    """ Run inference on images in img_path and log results in out_path """
    jpg_images = [
        filename.replace('jpg', '{}')
        for filename in os.listdir(img_path)
        if filename.split('.')[1] == 'jpg'
    ]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    for filename in jpg_images:
        input_filename = os.path.join(img_path, filename.format('jpg'))
        output_filename = os.path.join(out_path, filename.format('txt'))
        
        with open(output_filename, 'w') as out:
            img = cv2.imread(input_filename)
            detections = model.predict(img)
            detections = rescale_bbs(img, detections)

            for box in detections:
                img_cls = model.classes[box[6]]
                if (img_cls == 'boat'):
                    box_conf = box[4]
                    out.write('{0} {1} {2} {3} {4} {5}\n'.format(img_cls, box_conf, box[0], box[1], box[2], box[3]))
            
            if show_boxes:
                img_bb = plot_boxes_cv2(img, detections, detections, (0,0,255), class_names=model.classes)
                cv2.imshow('test', img_bb)
                cv2.waitKey(0)
        print('Processing {0}'.format(input_filename))

if __name__ == "__main__":
    classes = load_classes('config/classes.names')
    model = TrtDetector('single_cls.trt', ['boat'], 416, conf_thresh=0.3, nms_thresh=0.45)
    # Create detection results
    run_inference(model, '/home/johan/fiftyone/coco-2017/validation/data', 'detection-results')
