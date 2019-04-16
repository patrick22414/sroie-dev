import torch
import numpy
from PIL import Image, ImageDraw


RESO_H = 768
RESO_W = RESO_H // 2
GRID_H = 16
GRID_W = RESO_W // 5


def transform_pred(pred, ratio, conf_thresh):
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

    return conf, coor


def nms_filter(confidence, coordinate, iou_thresh):
    conf, indices = torch.sort(confidence, descending=True)
    coor = coordinate[indices]

    count = 1
    coor_nms = torch.zeros_like(coor)
    coor_nms[0] = coor[0]
    for i in range(1, len(coor)):
        if (calc_ious(coor_nms[0:count], coor[i]) < iou_thresh).all():
            coor_nms[count] = coor[i]
            count += 1

    return coor_nms[0:count]


def calc_f1(pred, truth, iou_thresh):
    ious = torch.zeros(len(pred), len(truth))
    for i, pred_box in enumerate(pred):
        ious[i] = calc_ious(truth, pred_box)
    ious_max, _ = torch.max(ious, dim=1)
    binary = (ious_max > iou_thresh).long()
    precisions = [binary[0:i].sum().item() / (i + 1) for i in range(len(binary))]
    precisions = [max(precisions[i:]) for i in range(len(precisions))]
    recalls = [binary[0:i].sum().item() / len(truth) for i in range(len(binary))]
    f1 = [2 * p * r / (p + r) for p, r in zip(precisions, recalls)]
    return f1, precisions, recalls


def calc_ious(boxes, box0):
    m = torch.max(boxes, box0).t()
    n = torch.min(boxes, box0).t()
    ious = ((n[2] - m[0]).clamp(0) * (n[3] - m[1]).clamp(0)) / (
        (m[2] - n[0]).clamp(0) * (m[3] - n[1]).clamp(0)
    )

    return ious


if __name__ == "__main__":
    torch.set_printoptions(threshold=10, edgeitems=2)
    # torch.manual_seed(1)
    image = Image.new("RGB", (RESO_W, RESO_H))
    pred = torch.stack(
        (
            torch.rand(RESO_H // GRID_H, RESO_W // GRID_W),
            torch.rand(RESO_H // GRID_H, RESO_W // GRID_W) - 0.5,
            torch.rand(RESO_H // GRID_H, RESO_W // GRID_W) - 0.5,
            torch.rand(RESO_H // GRID_H, RESO_W // GRID_W) + 0.5,
            torch.rand(RESO_H // GRID_H, RESO_W // GRID_W) + 0.5,
        )
    )
    conf, coor = transform_pred(pred, (1, 1), conf_thresh=0.5)
    nms_coor = nms_filter(conf, coor, iou_thresh=0.1)

    conf, indices = torch.sort(conf, descending=True)
    coor = coor[indices]

    draw = ImageDraw.Draw(image)
    for c in nms_coor:
        draw.rectangle(tuple(c), fill="white")
    for i, c in enumerate(coor):
        draw.rectangle(tuple(c), outline="red", width=1)
        draw.text(tuple(c[0:2]), f"{i}", fill="red")
    image.save("tmp.png")
