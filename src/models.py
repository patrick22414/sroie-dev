import time

import numpy
import torch

RESO_H = 768
RESO_W = RESO_H // 2
GRID_H = 16
GRID_W = RESO_W // 5


class LineModel(torch.nn.Module):
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

    def forward(self, inpt):
        features = self.feature_extractor(inpt)

        lines = self.line_detector(features).squeeze(dim=3)

        if self.training:
            confidence = lines[:, 0, :]
        else:
            confidence = torch.sigmoid(lines[:, 0, :])
        offset = torch.nn.functional.softsign(lines[:, 1, :])
        scaling = torch.nn.functional.softplus(lines[:, 2, :])

        return torch.stack([confidence, offset, scaling], dim=2)


class GridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            # 3 x 768 x 384
            Conv(3, 15, 5, padding=2),
            # 15 x 768 x 384
            Residual(15),
            ResidualReduce(15, 30),
            # 30 x 384 x 192
            Residual(30),
            ResidualReduce(30, 60),
            # 60 x 192 x 96
            Residual(60),
            ResidualReduce(60, 120),
            # 120 x 96 x 48
            Residual(120),
            ResidualReduce(120, 240),
            # 240 x 48 x 24
        )
        self.grid_detector = torch.nn.Sequential(
            # 240 x 48 x 24
            Conv(240, 270, (1, 3), padding=0, groups=5),
            # 270 x 48 x 22
            Conv(270, 300, (1, 3), padding=0, groups=5),
            # 300 x 48 x 20
            torch.nn.Conv2d(300, 300, (1, 4), stride=(1, 4), groups=5),
            # 300 x 48 x 5
            torch.nn.Conv2d(300, 5, 1, groups=5),
            # 5 x 48 x 5
        )

    def forward(self, input):
        features = self.feature_extractor(input)

        grids = self.grid_detector(features)

        if self.training:
            confidence = grids[:, 0, :, :]
        else:
            confidence = torch.sigmoid(grids[:, 0, :, :])
        offset_x = grids[:, 1, :, :]
        offset_y = grids[:, 2, :, :]
        scale_x = torch.nn.functional.softplus(grids[:, 3, :, :])
        scale_y = torch.nn.functional.softplus(grids[:, 4, :, :])

        return torch.stack([confidence, offset_x, offset_y, scale_x, scale_y], dim=1)


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
    numpy.set_printoptions(precision=2, suppress=True)

    model = LineModel()
    model.train()

    data = torch.randn(2, 3, RESO_H, RESO_W)

    start = time.time()
    pred = model(data)

    print("T:", time.time() - start)
    print(pred.size())
    print(pred.max().item(), pred.min().item())
