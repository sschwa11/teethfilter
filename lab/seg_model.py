import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class SegNet(nn.Module):
    def __init__(self, w=16, cout=1):
        super().__init__()
        c = [w, 2 * w, 4 * w, 8 * w, 8 * w]
        self.enc = nn.ModuleList()
        ci = 3
        for co in c:
            self.enc.append(nn.Sequential(conv(ci, co), conv(co, co)))
            ci = co
        self.dec = nn.ModuleList()
        for i in range(len(c) - 1, 0, -1):
            self.dec.append(nn.Sequential(conv(c[i] + c[i - 1], c[i - 1]), conv(c[i - 1], c[i - 1])))
        self.head = nn.Conv2d(c[0], cout, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips[-1]
        for j, d in enumerate(self.dec):
            s = skips[-2 - j]
            x = d(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return self.head(x)
