import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "..")
import dental as D

key = sys.argv[1]
opts = {k: eval(v) for k, v in (a.split("=") for a in sys.argv[2:])}
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
cap = cv2.VideoCapture(str(d["path"]))
af = D.AnchorFill()
af.reset(json.load(open("../settings.json"))["anchor_offset"])
af.load_state()
af.debug = True
for k, v in opts.items():
    setattr(af, k, v)
orig_measure = af.et.measure
last = {}


def spy(L, mouth):
    m = orig_measure(L, mouth)
    last["m"] = m
    return m


af.et.measure = spy
slot = af.slot > 0.5
recs = []
n = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    last.clear()
    af.dbg = None
    if d["ok"][n]:
        p = d["pts"][n].astype(np.float64)
        s, r, t = af.df.pose(p)
        pitch = float(np.degrees(np.arctan2(r[2, 1], r[2, 2])))
        yaw = float(np.degrees(np.arcsin(-np.clip(r[2, 0], -1, 1))))
        af.process(f, p)
        g = af.dbg
        if g is not None:
            L = cv2.cvtColor(g["crop"], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
            vis = af.donor_mask & (g["mouth"] > 0.7)
            ref = float(np.percentile(L[vis], 85)) if vis.sum() >= 20 else 0.0
            donor_bright = int((vis & (L > 0.62 * ref)).sum()) if ref > 0 else 0
            au = float(g["alpha_u"][slot].sum())
            m = last.get("m")
            my = (m["y"] - af.et.b0 - af.et.off[1]) if (m and m["y"] is not None) else np.nan
            mconf = m["conf"] if m else np.nan
            recs.append((n, pitch, yaw, donor_bright, au, af.delta[0], af.delta[1], my, mconf))
    n += 1
a = np.array(recs, dtype=np.float64)
np.save(f"out/pose_{key}.npy", a)
print(key, "frames", len(a), "pitch p5/p50/p95", np.round(np.percentile(a[:, 1], [5, 50, 95]), 1))
for q in (0, 1, 2, 3):
    lo, hi = np.percentile(a[:, 1], [q * 25, (q + 1) * 25])
    sel = (a[:, 1] >= lo) & (a[:, 1] <= hi) & (a[:, 3] >= 40)
    if sel.sum():
        ratio = a[sel, 4] / np.maximum(a[sel, 3], 1)
        print(f"  pitch {lo:6.1f}..{hi:6.1f}: frames with visible real tooth {int(sel.sum()):4d}  fill/real-tooth area median {np.median(ratio):.2f}  fill<0.3x {np.mean(ratio < 0.3):.2f}  delta_y median {np.median(a[sel, 6]):.1f}  measured incisal y offset median {np.nanmedian(a[sel, 7]):.1f}")
