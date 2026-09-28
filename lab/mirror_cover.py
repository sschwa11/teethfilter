import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "..")
import dental as D


def series(key, opts):
    d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
    cap = cv2.VideoCapture(str(d["path"]))
    af = D.AnchorFill()
    af.reset(json.load(open("../settings.json"))["anchor_offset"])
    af.load_state()
    af.debug = True
    for k, v in opts.items():
        setattr(af, k, v)
    xm = int(round(af.x_mid))
    res = []
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        af.dbg = None
        if d["ok"][n]:
            af.process(f, d["pts"][n].astype(np.float64))
        g = af.dbg
        if g is None or af.locks < 3:
            res.append(np.nan)
            n += 1
            continue
        L = cv2.cvtColor(g["crop"], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        real = L[:, xm + 4:xm + 31]
        m = g["mouth"][:, xm + 4:xm + 31] > 0.5
        zone = np.arange(D.CH)[:, None] <= af.bottom_full[xm + 4:xm + 31][None, :] + 2
        tooth = m & zone
        ref = float(np.percentile(real[tooth], 85)) if tooth.sum() >= 30 else 0.0
        if ref < 60:
            res.append(np.nan)
            n += 1
            continue
        t = (real >= 0.62 * ref) & zone & (g["mouth"][:, xm + 4:xm + 31] > 0.2)
        a = g["alpha"][:, xm - 30:xm - 3][:, ::-1]
        res.append(float(a[t].mean()) if t.sum() >= 30 else np.nan)
        n += 1
    return np.array(res)


if __name__ == "__main__":
    key = sys.argv[1]
    for arg in sys.argv[2:] or ["default"]:
        opts = {} if arg == "default" else {k: eval(v) for k, v in (a.split("=") for a in arg.split(","))}
        s = series(key, opts)
        import os
        r = os.environ.get("RANGE")
        if r:
            a0, a1 = (int(v) for v in r.split("-"))
            s = s[a0:a1 + 1]
        ok = np.isfinite(s)
        print(f"{key[:19]} {arg}: frames {int(ok.sum())}  mean cover of mirrored real tooth {np.nanmean(s):.3f}  frames <0.5: {int((s[ok] < 0.5).sum())}  <0.7: {int((s[ok] < 0.7).sum())}", flush=True)
