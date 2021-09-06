from utils import *
from scipy.optimize import linear_sum_assignment

import numpy as np

import random
import colorsys

class BoundingBoxTracker:

    def __init__(self, min_iou=0.3):
        self.tracking_list = []
        self.color_list = []
        self.min_iou = min_iou

    def track(self, detection_list):
        """ Track bounding boxes between frames.
            @param detection_list: List of bounding boxes 
        """
        # No boxes to track
        if (len(detection_list) == 0):
            return

        # At the beginning we have no t-1
        if (len(detection_list) > 0 and len(self.tracking_list) == 0):
            self.tracking_list = np.copy(detection_list)
            for tracker in self.tracking_list:
                h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
                self.color_list.append([int(256*i) for i in colorsys.hls_to_rgb(h,l,s)])

        iou_matrix_filtered = self.calculate_IOU_scores(detection_list)
        matched_idx = np.asarray(linear_sum_assignment(-iou_matrix_filtered))
        
        unmatched_trackers, unmatched_detections = [], []
        for t, tracker in enumerate(self.tracking_list):
            if (t not in matched_idx[1,:]):
                unmatched_trackers.append(t)

        for d, detection in enumerate(detection_list):
            if (d not in matched_idx[0,:]):
                unmatched_detections.append(d)
        
        # Update the color lists for new and matched detections
        new_color_list = [(0,0,0)] * len(detection_list)
        for unmatched in unmatched_detections:
            # Want light random colours
            h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
            new_color_list[unmatched] = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
        # Reuse colour for matches
        for i, tracker in enumerate(matched_idx[1,:]):
            new_color_list[matched_idx[0,i]] = self.color_list[tracker]
        
        self.color_list = new_color_list
        self.tracking_list = np.copy(detection_list)

    def calculate_IOU_scores(self, detection_list):
        """ Calculate iou matrix for trackers and detections
            @param detection_list: List of bounding boxes 
            @return iou_matrix_filtered
                - iou_matrix_filtered:  Matrix of iou scores
        """
        iou_matrix = np.zeros((len(detection_list), len(self.tracking_list)))
        for i, detection in enumerate(detection_list):
            for j, tracker in enumerate(self.tracking_list):
                iou = bbox_iou(detection, tracker, x1y1x2y2=False)
                iou_matrix[i,j] = iou
        
        iou_matrix_filtered = np.zeros((len(detection_list), len(self.tracking_list)))
        max_indeces = np.argmax(iou_matrix, axis=1)
        for i, j in enumerate(max_indeces):
            iou_matrix_filtered[i, j] = 1
        return iou_matrix_filtered

            

        