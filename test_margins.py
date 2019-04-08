import os
import glob
import numpy
import torch
from PIL import Image

from src import LineModel, draw_pred_line

DATA_PATH = "../sroie-data/"
RESO_H = 768
RESO_w = RESO_H // 2
GRID_H = 16


def test():
    filenames = [os.path.splitext(f)[0] for f in glob.glob(DATA_PATH + "data_tmp/*.jpg")]
    samples = random.sample(filenames, batch_size)
    jpg_files = [s + ".jpg" for s in samples]
    txt_files = [s + ".txt" for s in samples]

    # convert jpg files to NCWH tensor
    data = numpy.zeros([batch_size, 3, RESO_H, RESO_W], dtype=numpy.float32)
    ratio = numpy.zeros(batch_size)
    for i, f in enumerate(jpg_files):
        im = Image.open(f).convert("RGB")
        ratio[i] = RESO_H / im.height
        im = im.resize([RESO_W, RESO_H])
        data[i] = numpy.moveaxis(numpy.array(im), 2, 0)

    truth = numpy.zeros([batch_size, RESO_H // GRID_H, 3], dtype=numpy.float32)
    for i, (f, r) in enumerate(zip(txt_files, ratio)):
        truth[i] = txt_to_truth(f, r)

    return torch.tensor(data, device=device), torch.tensor(truth, device=device)


if __name__ == "__main__":
