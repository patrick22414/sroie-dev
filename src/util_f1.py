import numpy
import torch


def calc_f1(pred, truth, iou_thresh=0.5):
    """
    summary:
        calculates the F1, AP, AR of a single image

    arguments:
        pred    prediction boxes as an N x 4 tensor, each row is (x0, y0, x1, y1) of a box.
                prediction boxes are assumed to be in descending confidence, i.e. the first box has
                the highest confidence
        truth   truth boxes as an M x 4 tensor, each row is (x0, y0, x1, y1) of a box. the order of
                truth boxes does not matter
        iou_thresh  optional. the IOU required for a prediction box to be considered as 'correct'

    return:
        f1-score, average-precision, average-recall

    note:
        this does NOT work on CUDA, put things on CPU
    """
    ious = torch.zeros(len(pred), len(truth))
    for i, pred_box in enumerate(pred):
        ious[i] = calc_ious(truth, pred_box)
    ious_max, _ = torch.max(ious, dim=1)

    hit = (ious_max > iou_thresh).long().numpy()
    for i in range(len(hit)):
        hit[i] = hit[0:i].sum()

    precision = hit / numpy.arange(1, len(hit) + 1)
    for i in range(len(precision)):
        precision[i] = precision[i:].max()

    recall = hit / len(truth)

    # Average precision
    ap = numpy.interp(numpy.linspace(min(recall), max(recall), 101), recall, precision).mean()
    # Average recall
    ar = numpy.interp(numpy.linspace(min(precision), max(precision), 101), precision, recall).mean()
    # Average F1
    f1 = 2 * ap * ar / (ap + ar)

    return f1, ap, ar


def calc_ious(boxes, box0):
    m = torch.max(boxes, box0).t()
    n = torch.min(boxes, box0).t()
    ious = ((n[2] - m[0]).clamp(0) * (n[3] - m[1]).clamp(0)) / (
        (m[2] - n[0]).clamp(0) * (m[3] - n[1]).clamp(0)
    )

    return ious
