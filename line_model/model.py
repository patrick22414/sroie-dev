import torch
from torch import nn
from torch.nn import Module
import string

VALID_CHARS = "\r" + string.digits + string.ascii_uppercase + string.punctuation + " \n"

class LineEncoder(Module):
    pass


class LineDecoder(Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(VALID_CHARS), 18)


if __name__ == "__main__":
    pass
