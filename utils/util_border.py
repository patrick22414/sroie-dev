import os
from PIL import Image
import numpy
from skimage.filters import threshold_local
from skimage.morphology import dilation, erosion
from matplotlib import pyplot


def test():
    files = [f.path for f in os.scandir("../sroie-data/task1/data_bordered/") if f.name.endswith(".jpg")]

    for f in files:
        print(f)
        im = numpy.array(Image.open(f).convert("L"))
        im_bin = threshold_local(im, block_size=9).astype(numpy.uint8)
        # print(im_bin)
        Image.fromarray(im_bin).save(os.path.splitext(f)[0] + "-bin.png")
        # break


if __name__ == "__main__":
    test()
