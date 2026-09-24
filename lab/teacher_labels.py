import os
import sys
import time

import cv2
import numpy as np

import common as C
import parsers as P


def run(keys, step=2):
    df = C.dental()
    ps = {k: P.Parser(k) for k in ("segformer", "bisenet")}
    for key in keys:
        out = os.path.join(C.CACHE, key + "_teach.npz")
        if os.path.exists(out):
            continue
        d = np.load(os.path.join(C.CACHE, key + ".npz"), allow_pickle=True)
        cap = cv2.VideoCapture(str(d["path"]))
        n = 0
        t0 = time.time()
        labels = np.zeros((len(d["ok"]), 2, 3, C.CH, C.CW), np.uint8)
        crops = np.zeros((len(d["ok"]), C.CH, C.CW, 3), np.uint8)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if d["ok"][n] and n % step == 0:
                pts = d["pts"][n].astype(np.float64)
                a = df.affine(df.pose(pts))
                crops[n] = cv2.warpAffine(f, a.astype(np.float32), (C.CW, C.CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                for j, name in enumerate(("segformer", "bisenet")):
                    p, m, _ = ps[name].probs(f, pts)
                    dd = ps[name].to_dental(p, m, a)
                    for c, cls in enumerate(("mouth", "u_lip", "l_lip")):
                        labels[n, j, c] = np.clip(dd[cls] * 255, 0, 255).astype(np.uint8)
            n += 1
        sel = (np.arange(n) % step == 0) & d["ok"][:n]
        np.savez_compressed(out, labels=labels[:n], crops=crops[:n], ok=sel)
        print(f"{key}: {n} frames in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    order = ["2026-09-22_23-02-18", "2026-09-23_05-46-00_raw", "2026-09-23_05-47-41_raw", "2026-09-22_23-36-22_raw", "2026-09-23_05-46-26_raw", "2026-09-23_05-46-53_raw", "2026-09-23_05-47-17_raw", "2026-09-22_23-36-51_raw"]
    run(sys.argv[1:] or order + [k for k in C.keys() if k not in order])
