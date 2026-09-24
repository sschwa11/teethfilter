import os
import sys
import time

import numpy as np
import torch

import common as C
from seg_model import SegNet


class Wrapped(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)[:, [2, 1, 0]] / 255.0
        return torch.sigmoid(self.net(x))[:, 0]


def main(name="seg_w16", width=16, out=None):
    net = SegNet(width)
    net.load_state_dict(torch.load(os.path.join(C.CACHE, name + ".pt"), map_location="cpu"))
    net.eval()
    w = Wrapped(net)
    dummy = torch.zeros(1, 256, 320, 3)
    out = out or os.path.join(C.CACHE, name + ".onnx")
    torch.onnx.export(w, dummy, out, input_names=["bgr"], output_names=["mouth"], opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    x = np.random.randint(0, 255, (1, 256, 320, 3)).astype(np.float32)
    for _ in range(5):
        sess.run(None, {"bgr": x})
    t = time.perf_counter()
    for _ in range(50):
        y = sess.run(None, {"bgr": x})[0]
    print(out, sess.get_providers()[0], f"{(time.perf_counter() - t) / 50 * 1000:.2f} ms", y.shape)
    with torch.no_grad():
        ref = w(torch.from_numpy(x)).numpy()
    print("max abs diff vs torch", float(np.abs(ref - y).max()))


if __name__ == "__main__":
    main(*sys.argv[1:2])
