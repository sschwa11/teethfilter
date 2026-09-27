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
prev = None
prev_opn = None
changes = []
n = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    af.dbg = None
    cur = None
    opn = None
    if d["ok"][n]:
        p = d["pts"][n]
        af.process(f, p.astype(np.float64))
        g = af.dbg
        if g is not None:
            cur = g["alpha_u"] > 0.5
            opn = float(np.linalg.norm(p[13, :2] - p[14, :2]) / max(np.linalg.norm(p[78, :2] - p[308, :2]), 1))
    if cur is not None and prev is not None and cur.sum() + prev.sum() > 300 and abs(opn - prev_opn) < 0.015:
        inter = (cur & prev).sum()
        union = (cur | prev).sum()
        changes.append((n, 1 - inter / max(union, 1)))
    prev, prev_opn = cur, opn
    n += 1
c = np.array([x[1] for x in changes])
big = [x for x in changes if x[1] > 0.3]
print("big:", [(int(a), round(float(b), 2)) for a, b in big][:40])
print(f"{key[:19]} {' '.join(sys.argv[2:]) or 'default'}: steady-lip frame pairs {len(c)}, mean shape change {c.mean():.3f}, pairs >15% {int((c > 0.15).sum())}, >30% {int((c > 0.3).sum())}")
