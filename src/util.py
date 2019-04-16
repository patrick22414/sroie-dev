import torch
import string
import glob
import os
import pickle
from PIL import Image, ImageDraw, ImageFont
from sys import maxsize

VALID_CHARS = string.ascii_uppercase + string.digits + string.punctuation + " "

#Red -- Orange -- Yellow -- Green -- Blue -- Cyan -- and Violet
RAINBOW = ["red", "gold", "orange", "green", "blue", "violet", "purple"]

class TextLine(object):
    def __init__(self):
        self.text = []
        self.xs = []
        self.yspan = None

    def insert(self, x, text, yspan):
        try:
            at = next(i for i, v in enumerate(self.xs) if v > x)
        except StopIteration:
            self.text.append(text)
            self.xs.append(x)
        else:
            self.text.insert(at, text)
            self.xs.insert(at, x)

        if self.yspan is None:
            self.yspan = list(yspan)
        else:
            if yspan[0] < self.yspan[0]:
                self.yspan[0] = yspan[0]
            if yspan[1] > self.yspan[1]:
                self.yspan[1] = yspan[1]

    def __repr__(self):
        if self.yspan is None:
            repr_yspan = "[    ,    ] "
        else:
            repr_yspan = "[{:4d},{:4d}] ".format(self.yspan[0], self.yspan[1])

        repr_text = "\t".join(self.text)

        return repr_yspan + repr_text


def check_char():
    txt_files = glob.glob("../task1/*.txt")
    for txt in txt_files:
        with open(txt, "r", encoding="utf-8") as lines:
            for line in lines:
                for c in line.strip().split(",")[-1]:
                    if not c in VALID_CHARS:
                        print(txt, line, c)


def reorg_txt():
    txt_files = glob.glob("../task1/*.txt")
    for txt in txt_files:
        with open(txt, "r", encoding="utf-8") as lines:
            new_lines = [None] * sum(1 for l in lines)
            lines.seek(0)
            for i, line in enumerate(lines):
                new_line = line.strip().split(",", maxsplit=8)
                new_line = (
                    tuple(int(n) for n in new_line[0:2] + new_line[4:6]),
                    new_line[-1],
                )
                new_lines[i] = new_line
        with open(os.path.splitext(txt)[0] + ".pickle", "wb") as pickle_file:
            pickle.dump(new_lines, pickle_file)


def batch_rename():
    filenames = [
        os.path.splitext(f)[0]
        for f in os.scandir("../task1/")
        if f.name.endswith(".jpg")
    ]
    for i, f in enumerate(filenames, 1):
        os.rename(f + ".jpg", "../task1/{:03d}".format(i) + ".jpg")
        os.rename(f + ".txt", "../task1/{:03d}".format(i) + ".txt")


def line_print():
    jpg_files = [f.path for f in os.scandir("../sroie-data/task1/") if f.name.endswith(".jpg")]
    pck_files = [os.path.splitext(f)[0] + ".pickle" for f in jpg_files]

    font = ImageFont.truetype("Inconsolata-Regular.ttf", size=12)

    for f_jpg, f_pck in zip(jpg_files, pck_files):
        image = Image.open(f_jpg).convert("RGB")

        with open(f_pck, "rb") as f_pck_opened:
            labels = pickle.load(f_pck_opened)

        text_lines = [TextLine() for _ in range(768 // 16)]
        for coor, text in labels:
            x = (coor[0] + coor[2]) / 2 * (384 / image.width)
            y = (coor[1] + coor[3]) / 2 * (768 / image.height)
            try:
                text_lines[int(y / 16)].insert(x, text, (coor[1], coor[3]))
            except IndexError as e:
                print(y, int(y / 16))
                raise e

        draw = ImageDraw.Draw(image)
        color_id = 0
        for line in text_lines:
            if len(line.text) != 0:
                color = RAINBOW[color_id % len(RAINBOW)]
                draw.line((0, line.yspan[0], image.width, line.yspan[0]), fill=color, width=1)
                draw.line((0, line.yspan[1], image.width, line.yspan[1]), fill=color, width=2)
                draw.text((10, line.yspan[0] + 2), "\t".join(line.text), fill=color, font=font)
                color_id += 1
        image.save(os.path.splitext(f_jpg)[0] + ".png")
        break

        print(f_jpg)


def collect_char():
    pck_files = [f.path for f in os.scandir("../sroie-data/task1/") if f.name.endswith(".pickle")]

    collection = set()
    for f in pck_files:
        print(f)
        with open(f, "rb") as f_opened:
            labels = pickle.load(f_opened)

        for _, text in labels:
            collection.update(text)

    print(len(collection))
    print(collection)


if __name__ == "__main__":
    line_print()
