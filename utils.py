import torch
import random
import colorsys
import numpy as np
import cv2

def _true_divide(dividend, divisor):
    return torch.true_divide(dividend, divisor)

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    with open(path, 'r') as f:
        names = f.read().split("\n")
        # Filter removes empty strings (such as last line)
        return list(filter(None, names))  

def pre_process(img, img_size):
    """ Perform pre processing on an image
        @param img:          Image to pre process
        @param img_size:     Size of the image
        @return bboxes_batch
            - img_in:  Preprocessed image
    """
    resized = cv2.resize(
        img, (img_size, img_size), interpolation=cv2.INTER_LINEAR
    )
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    img_in = np.ascontiguousarray(img_in)
    return img_in

def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    """ Perform non-maximum supression (NMS) """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)

def bbox_iou(box1, box2, x1y1x2y2=True):
    """ Method to calculate intersection over union of two bbs. """
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

        mx = min(box1[0], box2[0])
        Mx = max(box1[0] + w1, box2[0] + w2)
        my = min(box1[1], box2[1])
        My = max(box1[1] + h1, box2[1] + h2)
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def post_processing(conf_thresh, nms_thresh, output):
    """ Perform post processing on a batch
        @param conf_thres:   Threshold on the detection confidence
        @param nms_thres:    Threshold for Non Maximum Suppression
        @param output:       Bounding boxes output from inference
        @return bboxes_batch
            - bboxes_batch:  List of each post processed bounding boxes in the batch
    """
    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    if type(box_array).__name__ != "ndarray":
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    for batch in range(box_array.shape[0]):

        argwhere = max_conf[batch] > conf_thresh
        l_box_array = box_array[batch, argwhere, :]
        l_max_conf = max_conf[batch, argwhere]
        l_max_id = max_id[batch, argwhere]

        bboxes = []
        # nms for each class
        for cls_id in range(num_classes):

            cls_argwhere = l_max_id == cls_id
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if keep.size > 0:
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for box in range(ll_box_array.shape[0]):
                    bboxes.append(
                        [
                            ll_box_array[box, 0],
                            ll_box_array[box, 1],
                            ll_box_array[box, 2],
                            ll_box_array[box, 3],
                            ll_max_conf[box],
                            ll_max_conf[box],
                            ll_max_id[box],
                        ]
                    )

        bboxes_batch.append(bboxes)

    return bboxes_batch

def rescale_bbs(img, bbs):
    """ Rescale bounding boxes to image coordinates
        @param img:     Original image 
        @param bbs:     Bounding boxes to rescale
        @return bbs
            - bbs:  List of rescaled bounding boxes
    """
    width = img.shape[1]
    height = img.shape[0]
    for box in bbs:
        box[0] = int(box[0] * width)
        box[1] = int(box[1] * height)
        box[2] = int(box[2] * width)
        box[3] = int(box[3] * height)
    return bbs

def plot_boxes_cv2(img, boxes, color_list, savename=None, class_names=None):
    """ Plot boxes onto provided image
        @param img:         Image to plot boxes on
        @param boxes:       Boxes to be drawn
        @param savename:    Optional, path to save image
        @param class_names: List of the names of the classes
        @param savename:    Optional, path to save image
        @return img
            - img:  Image with bounding boxes
    """
    img = np.copy(img)

    n_boats = 0
    for box_id, box in enumerate(boxes):
        if len(box) >= 7 and class_names:
            x1 = box[0]
            y1 = box[1] 
            x2 = box[2]
            y2 = box[3]
            
            cls_id = box[6]
            area = (x2-x1) * (y2-y1)

            # Arbitary area number to remove self detection
            if cls_id == 8 and area <= 100000:
                # BGR color codes
                rgb = color_list[box_id]

                img = cv2.putText(
                    img,
                    class_names[cls_id],
                    (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    rgb,
                    1,
                    cv2.LINE_AA,
                )
                img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 2)
                n_boats += 1
    # Display number of boxes
    img = cv2.putText(
        img,
        'Number of boats: {0}'.format(n_boats),
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (180, 105, 255), # pink
        1,
        cv2.LINE_AA,
    )

    if savename:
        print("save plot results to {}".format(savename))
        cv2.imwrite(savename, img)
    return img