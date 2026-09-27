import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "..")
import dental as D

key = sys.argv[1]
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
cap = cv2.VideoCapture(str(d["path"]))
af = D.AnchorFill()
af.reset(json.load(open("../settings.json"))["anchor_offset"])
af.load_state()
af.debug = True
rows = np.arange(D.CH)[:, None]
xm = af.x_mid
up_edge = int(xm - 66)
up_band = (rows >= af.body_top[None, :]) & (rows <= af.bottom_full[None, :] + 1)
lo_band = rows > af.bottom_full[None, :] + 2
ext_u, ext_l = [], []
n = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    af.dbg = None
    if d["ok"][n]:
        af.process(f, d["pts"][n].astype(np.float64))
        g = af.dbg
        if g is not None:
            m = g["mouth"] > 0.5
            if g["alpha_u"].sum() > 150:
                cols = np.flatnonzero((m & up_band)[:, :up_edge].sum(0) >= 4)
                ext_u.append(up_edge - cols.min() if cols.size else 0)
            if g["alpha_l"].sum() > 150 and af.lax is not None:
                lo_edge = int(af.lax + af.l_span)
                cols = np.flatnonzero((m & lo_band)[:, lo_edge:].sum(0) >= 4)
                ext_l.append(cols.max() + 1 if cols.size else 0)
    n += 1
eu, el = np.array(ext_u), np.array(ext_l)
print(f"{key[:19]}  upper fill frames {len(eu)}: opening reaches >=15 px past the fill {np.mean(eu >= 15):.0%}, >=30 px {np.mean(eu >= 30):.0%}  |  "
      f"lower fill frames {len(el)}: >=15 px past {np.mean(el >= 15):.0%}, >=30 px {np.mean(el >= 30):.0%}")
