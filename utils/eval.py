from __future__ import division, print_function

import numpy as np
import time

# TODO: Rewrite the accuracy calculation part to output recording
def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_acc(pred_bboxes_batch, target_bboxes_batch):
    iou_threshold = 0.5
    frame_num = len(pred_bboxes_batch)
    false_positives = np.zeros((0,))
    true_positives = np.zeros((0,))
    num_gt_annos = 0
    for i in range(frame_num):
        detected_annotations = []
        pred_bboxes, target_bboxes = pred_bboxes_batch[i], target_bboxes_batch[i]
        pred_bboxes = pred_bboxes.data.cpu().numpy()
        target_bboxes = np.array(target_bboxes)
        if target_bboxes.size == 0:
            # no ground truth bbox in this frame
            continue
        num_gt_annos += target_bboxes.size
        overlaps = compute_overlap(pred_bboxes, target_bboxes)
    
        assigned_bboxes = np.argmax(overlaps, axis=1)

        for j in range(assigned_bboxes.size):
            # in total N predicted bboxes
            assigned_bbox = assigned_bboxes[j]
            max_overlap = overlaps[j][assigned_bbox]
            if max_overlap >= iou_threshold and assigned_bbox not in detected_annotations:
                false_positives = np.append(false_positives, 0)
                true_positives = np.append(true_positives, 1)
                detected_annotations.append(assigned_bbox)
            else:
                false_positives = np.append(false_positives, 1)
                true_positives = np.append(true_positives, 0)

    if false_positives.size == 0:
        return -1.0

    false_positives = np.cumsum(false_positives)
    true_positives  = np.cumsum(true_positives)
    recall = true_positives / num_gt_annos
    precision = true_positives / (true_positives + false_positives)
    average_precision  = _compute_ap(recall, precision)

    return average_precision

def bbox_ious(pred_bboxes_batch, target_bboxes_batch):
    frame_num = len(pred_bboxes_batch)
    total_ious = 0.0
    total_bbox = 0
    for i in range(frame_num):
        detected_annotations = []
        pred_bboxes, target_bboxes = pred_bboxes_batch[i], target_bboxes_batch[i]
        pred_bboxes = pred_bboxes.data.cpu().numpy()
        target_bboxes = np.array(target_bboxes)
        if target_bboxes.size == 0:
            # no ground truth bbox in this frame
            continue

        overlaps = compute_overlap(pred_bboxes, target_bboxes)
    
        assigned_bboxes = np.argmax(overlaps, axis=1)

        for j in range(assigned_bboxes.size):
            # in total N predicted bboxes
            assigned_bbox = assigned_bboxes[j]
            max_overlap = overlaps[j][assigned_bbox]
            total_ious += max_overlap
            total_bbox += 1

    return total_ious / total_bbox
