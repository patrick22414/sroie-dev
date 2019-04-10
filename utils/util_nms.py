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
        [torch.arange(GRID_W // 2, RESO_W, GRID_W).float().to(pred.device)] * (RESO_H // GRID_H),
        dim=0,
    )
    center_y = torch.stack(
        [torch.arange(GRID_H // 2, RESO_H, GRID_H).float().to(pred.device)] * (RESO_W // GRID_W),
        dim=1,
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
        dim=1,
    )

    conf, indices = torch.sort(conf, descending=True)
    coor = coor[indices]
    i = 0
    while i < len(coor) - 1:
        coor = coor[(calc_ious(coor[i + 1 :], coor[i]) < iou_thresh).nonzero().squeeze()]
        i += 1

    return coor


def calc_ious(boxes, box0):
    m = torch.max(boxes, box0).t()
    n = torch.min(boxes, box0).t()
    ious = ((n[2] - m[0]) * (n[3] - m[1])) / ((m[2] - n[0]) * (m[3] - n[1]))
    return ious


if __name__ == "__main__":
    x = torch.arange(5 * 2).reshape(5, 2)
    y = torch.tensor([0, 1, 1, 0, 1])

    print(x)
    print(y.nonzero().squeeze())
    print(x[y.nonzero().squeeze()])
