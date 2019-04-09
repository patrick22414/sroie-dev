import os
import torch
import numpy
import string

VALID_CHARS = "\r" + string.digits + string.ascii_uppercase + string.punctuation + " \n"


def test():
    pickle_files = [
        f.path for f in os.scandir("../sroie-data/task1/") if f.name.endswith(".pickle")
    ]


def string_to_array(s, vocab=VALID_CHARS):
    array = numpy.zeros((len(s), len(vocab)))
    for i, c in enumerate(s):
        array[i, vocab.find(c)] = 1
    
    return array


if __name__ == "__main__":
    s1 = "\rLorem\n".upper()
    print(string_to_array(s1))
