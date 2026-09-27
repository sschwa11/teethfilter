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
slot = af.slot > 0.5
recs = []
n = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    af.dbg = None
    if d["ok"][n]:
        p = d["pts"][n]
        af.process(f, p.astype(np.float64))
        g = af.dbg
        if g is not None:
            opn = float(np.linalg.norm(p[13, :2] - p[14, :2]) / max(np.linalg.norm(p[78, :2] - p[308, :2]), 1))
            au = float(g["alpha_u"][slot].sum())
            mo = float((g["mouth"][slot] > 0.5).sum())
            recs.append((n, opn, au, mo))
    n += 1
a = np.array(recs)
np.save(f"out/updrop_{key}.npy", a)
ev = []
for i in range(1, len(a) - 2):
    n, opn, au, mo = a[i]
    if a[i - 1][0] != n - 1:
        continue
    prev = a[i - 1][2]
    nxt = max(a[i + 1][2], a[i + 2][2])
    if prev > 150 and au < 0.6 * prev and nxt > 0.8 * prev and abs(opn - a[i - 1][1]) < 0.03:
        ev.append((int(n), round(prev), round(au), round(nxt), round(a[i - 1][3]), round(mo)))
print(key, "frames", len(a), "upper-fill dips (prev -> this -> recovered, mouth px prev -> this):", len(ev))
for e in ev[:40]:
    print("  frame", e[0], "alpha", e[1], "->", e[2], "->", e[3], "| mouth px", e[4], "->", e[5])
