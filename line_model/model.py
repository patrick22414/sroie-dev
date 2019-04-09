import torch
from torch import nn
from torch.nn import Module
import string

VALID_CHARS = "\r" + string.digits + string.ascii_uppercase + string.punctuation + " \n"


class ROIFinder(Module):
    def __init__(self):
        super().__init__()


class LineEncoder(Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            # 3 x 768 x 384
            Conv(3, 12, 5, padding=2),
            # 12 x 768 x 384
            Residual(12),
            ResidualReduce(12, 24),
            # 24 x 384 x 192
            Residual(24),
            ResidualReduce(24, 48),
            # 48 x 192 x 96
            Residual(48),
            ResidualReduce(48, 96),
            # 96 x 96 x 48
            Residual(96),
            ResidualReduce(96, 192),
            # 192 x 48 x 24
        )
        self.line_detector = torch.nn.Sequential(
            # 192 x 48 x 24
            Conv(192, 216, (1, 3), padding=0, groups=3),
            # 240 x 48 x 22
            Conv(216, 240, (1, 3), padding=0, groups=3),
            # 240 x 48 x 20
            torch.nn.Conv2d(240, 240, (1, RESO_H // 32 - 4), groups=3),
            # 240 x 48 x 1
            torch.nn.Conv2d(240, 3, 1, groups=3),
            # 3 x 48 x 1
        )


class LineDecoder(Module):
    def __init__(self):
        super().__init__()


class Residual(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Sequential(
            Conv(channels, channels, 3, padding=1),
            Conv(channels, channels, 1, padding=0),
        )

    def forward(self, inpt):
        oupt = self.conv(inpt)
        return oupt + inpt


class ResidualReduce(torch.nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.branch_0 = torch.nn.Sequential(
            Conv(in_chan, out_chan, 3, padding=1, stride=2),
            Conv(out_chan, out_chan, 1, padding=0),
        )
        self.branch_1 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            Conv(in_chan, out_chan, 1, padding=0),
        )

    def forward(self, inpt):
        oupt_0 = self.branch_0(inpt)
        oupt_1 = self.branch_1(inpt)
        return oupt_0 + oupt_1


class Conv(torch.nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, padding, stride=1, groups=1):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_chan, out_chan, kernel_size, stride, padding, groups=groups),
            torch.nn.BatchNorm2d(out_chan),
            torch.nn.LeakyReLU(inplace=True),
        )

    def forward(self, inpt):
        return self.conv(inpt)


if __name__ == "__main__":
    pass
