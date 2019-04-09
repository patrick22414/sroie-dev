import os
import torch
import numpy
import string
import random
import pickle
from PIL import Image


VALID_CHARS = "\r" + string.digits + string.ascii_uppercase + string.punctuation + " \n"


def get_roi_data(batch_size):
    jpg_files = [f.path for f in os.scandir("../sroie-data/task1/") if f.name.endswith(".jpg")]
    jpg_files = random.sample(jpg_files, batch_size)
    pck_files = [os.path.splitext(f)[0] + ".pickle" for f in jpg_files]

    print(jpg_files)

    data = numpy.zeros((batch_size, 1, 256, 256), dtype=numpy.float32)
    rxs = numpy.zeros(batch_size)
    rys = numpy.zeros(batch_size)
    for i, f in enumerate(jpg_files):
        im = Image.open(f).convert("L")
        rxs[i] = 256 / im.width
        rys[i] = 256 / im.height
        data[i] = numpy.array(im.resize((256, 256)))

    truth = numpy.zeros((batch_size, 4), dtype=numpy.float32)
    for i, f in enumerate(pck_files):
        with open(f, "rb") as f_opened:
            label = pickle.load(f_opened)
            coors = numpy.zeros((len(label), 4))
            for ii, (coor, _) in enumerate(label):
                coors[ii] = coor
            truth[i] = (
                numpy.min(coors[:, 0]) * rxs[i],
                numpy.min(coors[:, 1]) * rys[i],
                numpy.max(coors[:, 2]) * rxs[i],
                numpy.max(coors[:, 3]) * rys[i],
            )

    return data, truth


def string_to_array(s, vocab=VALID_CHARS):
    array = numpy.zeros((len(s), len(vocab)))
    for i, c in enumerate(s):
        array[i, vocab.find(c)] = 1

    return array


if __name__ == "__main__":
    print(get_roi_data(1))
