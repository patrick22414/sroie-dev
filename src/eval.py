import argparse

import torch
from matplotlib import pyplot
from PIL import ImageDraw

from data import get_all_eval_grid
from model import GridModel, RESO_W, RESO_H
from util_nms import calc_f1, nms_filter, transform_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-m", "--model")

    args = parser.parse_args()
    args.device = torch.device(args.device)

    model = GridModel()
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.eval()

    with torch.no_grad():
        data, truth, images = get_all_eval_grid()
        for i, (d, t, im) in enumerate(zip(data, truth, images)):
            pred = model(d).squeeze()
            conf, coor = transform_pred(pred, (RESO_W / im.width, RESO_H / im.height), conf_thresh=0.5)
            coor_nms = nms_filter(conf, coor, iou_thresh=0.1)
            # f1, precision, recall = calc_f1(coor_nms, t, iou_thresh=0.5)
            # print("F1 Score:  ", f1)
            # print("Precision: ", precision)
            # print("Recall:    ", recall)

            # pyplot.plot(recall, precision)
            draw = ImageDraw.Draw(im)
            for c in coor_nms:
                draw.rectangle(tuple(c), outline="magenta", width=2)
            im.save(f"{i}.png")


if __name__ == "__main__":
    main()
