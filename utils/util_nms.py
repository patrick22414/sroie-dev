import torch
import numpy

RESO_H = 768
RESO_W = RESO_H // 2
GRID_H = 16
GRID_W = RESO_W // 5


def transform_pred(pred, ratio, conf_thresh=0.5, iou_thresh=0.5):
    conf = pred[0]
    mask = conf > conf_thresh

    center_x = torch.stack(
        [torch.arange(GRID_W // 2, RESO_W, GRID_W).float().to(pred.device)] * (RESO_H // GRID_H), dim=0
    )
    center_y = torch.stack(
        [torch.arange(GRID_H // 2, RESO_H, GRID_H).float().to(pred.device)] * (RESO_W // GRID_W), dim=1
    )

    center_x = (pred[1] * GRID_W + center_x)[mask]
    center_y = (pred[2] * GRID_H + center_y)[mask]

    w_half = pred[3][mask] * GRID_W / 2
    h_half = pred[4][mask] * GRID_H / 2

    conf = conf[mask]
    coor = torch.stack(
        (
            (center_x - w_half) / ratio[0],
            (center_y - h_half) / ratio[1],
            (center_x + w_half) / ratio[0],
            (center_y + h_half) / ratio[1],
        ),
        dim = 1
    )
    # while True:
    conf, indices = torch.sort(conf, descending=True)
    coor = torch.index_select(coor, 0, indices)
    # iou_mask = calc_ious(coor[1:], coor[0]) < iou_thresh

    return conf, coor


def calc_ious(boxes, box0):
    pass


if __name__ == "__main__":
    pred = torch.rand(5, 48, 5)
    ratio = [0.8, 0.9]
    conf_thresh = 0.5
    coor = transform_pred(pred, ratio, conf_thresh)
    for c in coor:
        print(c)
