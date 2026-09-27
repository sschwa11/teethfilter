import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "..")
import dental as D

key = sys.argv[1]
paths = sys.argv[2:]
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
cap = cv2.VideoCapture(str(d["path"]))
caps = [cv2.VideoCapture(p) for p in paths]
af = D.AnchorFill()
af.reset(json.load(open("../settings.json"))["anchor_offset"])
af.debug = True
af.upper = False
af.lower = False
res = [[] for _ in paths]
frames = []
n = 0
while True:
    ok, f = cap.read()
    outs = [c.read()[1] for c in caps]
    if not ok or any(o is None for o in outs):
        break
    af.dbg = None
    if d["ok"][n]:
        af.process(f, d["pts"][n].astype(np.float64))
    g = af.dbg
    if g is not None and af.locks >= 3:
        L = cv2.cvtColor(g["crop"], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        twin = af.rescue_band(L, g["mouth"]) > 0.5
        if twin.sum() >= 40:
            xm = int(round(af.x_mid))
            donor = np.fliplr(twin)
            shift = int(round(2 * af.x_mid)) - (D.CW - 1)
            donor = np.roll(donor, shift, axis=1)
            ref = float(np.percentile(L[donor], 60)) if donor.sum() >= 20 else float(np.percentile(L[twin], 90))
            frames.append(n)
            for r, o in zip(res, outs):
                c = cv2.warpAffine(o, af.last[0].astype(np.float32), (D.CW, D.CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                Lo = cv2.cvtColor(c, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
                r.append(float((Lo[twin] >= 0.75 * ref).mean()))
    n += 1
print(key[:19], "frames where the real front tooth band is visible:", len(frames))
for p, r in zip(paths, res):
    r = np.array(r)
    if r.size:
        print(f"  {os.path.basename(os.path.dirname(p)) + '/' + os.path.basename(p):55s} twin drawn {r.mean():.2f}  twin missing (<50% bright) {np.mean(r < 0.5):.2f}  at {[fr for fr, v in zip(frames, r) if v < 0.5][:25]}")
