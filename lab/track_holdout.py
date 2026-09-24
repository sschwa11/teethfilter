import sys

import cv2
import numpy as np

import render_v10c as V
from run_v10c import MouthSource


def run(keys, configure=None, run_len=10, period=40):
    ms = MouthSource("seg_w16")
    model = V.Model()
    errs, meas, vis = [], 0, 0
    for key in keys:
        d = np.load(f"cache/{key}.npz", allow_pickle=True)
        cap = cv2.VideoCapture(str(d["path"]))
        r = V.Renderer(model, lower=False)
        if configure:
            configure(r)
        n = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if d["ok"][n]:
                pts = d["pts"][n].astype(np.float64)
                hide = (n % period) >= period - run_len
                r.process(f, pts, ms(f, pts), hide=hide)
                if r.miss == 0:
                    meas += 1
                if r.last_hidden is not None:
                    errs.append(abs(r.delta[0] - r.last_hidden[0]))
            n += 1
    e = np.array(errs)
    return {"hidden_n": len(e), "mae": float(e.mean()), "p90": float(np.percentile(e, 90)), "gt3": float((e > 3).mean()), "measured": meas}


if __name__ == "__main__":
    keys = sys.argv[1:] or ["2026-09-23_05-47-17_raw", "2026-09-23_05-46-00_raw", "2026-09-22_23-36-51_raw"]
    print(run(keys))
