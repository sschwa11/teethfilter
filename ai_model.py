import torch
import torch.nn as nn
import torch.nn.functional as F


def block(ci, co, stride=1, dil=1):
    return nn.Sequential(nn.Conv2d(ci, co, 3, stride, padding=dil, dilation=dil), nn.LeakyReLU(0.2, inplace=True))


class TeethNet(nn.Module):
    def __init__(self, w=32, cin=7):
        super().__init__()
        self.e1 = nn.Sequential(block(cin, w), block(w, w))
        self.e2 = nn.Sequential(block(w, 2 * w, 2), block(2 * w, 2 * w))
        self.e3 = nn.Sequential(block(2 * w, 4 * w, 2), block(4 * w, 4 * w))
        self.e4 = nn.Sequential(block(4 * w, 8 * w, 2), block(8 * w, 8 * w))
        self.mid = nn.Sequential(block(8 * w, 8 * w, dil=2), block(8 * w, 8 * w, dil=4), block(8 * w, 8 * w, dil=2))
        self.d3 = nn.Sequential(block(12 * w, 4 * w), block(4 * w, 4 * w))
        self.d2 = nn.Sequential(block(6 * w, 2 * w), block(2 * w, 2 * w))
        self.d1 = nn.Sequential(block(3 * w, w), block(w, w))
        self.out = nn.Conv2d(w, 3, 3, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(a)
        c = self.e3(b)
        d = self.mid(self.e4(c))
        u = self.d3(torch.cat([F.interpolate(d, scale_factor=2.0, mode="bilinear", align_corners=False), c], 1))
        u = self.d2(torch.cat([F.interpolate(u, scale_factor=2.0, mode="bilinear", align_corners=False), b], 1))
        u = self.d1(torch.cat([F.interpolate(u, scale_factor=2.0, mode="bilinear", align_corners=False), a], 1))
        m = x[:, 3:4]
        base = x[:, 4:7] * m + x[:, 0:3] * (1 - m)
        return torch.clamp(base + 0.25 * torch.tanh(self.out(u)), 0, 1)


def ai_input(y, m, hint):
    return torch.cat([y * (1 - m) + 0.5 * m, m, hint * m + 0.5 * (1 - m)], 1)
