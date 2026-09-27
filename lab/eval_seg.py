import os
import sys

import numpy as np
import torch

import common as C
from seg_model import SegNet
from train_seg import evaluate, load

keys = sys.argv[1].split(",")
names = sys.argv[2:]
dev = torch.device("cuda")
for name in names:
    net = SegNet(16).to(dev)
    net.load_state_dict(torch.load(os.path.join(C.CACHE, name + ".pt"), map_location=dev))
    net.eval()
    for k in keys:
        xv, yv = load([k])
        iou, mae, be = evaluate(net, xv, yv, dev)
        print(f"{name:10s} {k[:19]}  frames {len(xv):4d}  IoU {iou:.3f}  MAE {mae:.4f}  top-edge err {be:.2f}px", flush=True)
