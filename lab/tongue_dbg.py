import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "..")
import dental as D

key = sys.argv[1]
frames = [int(x) for x in sys.argv[2].split(",")]
out = sys.argv[3]
opts = {k: eval(v) for k, v in (a.split("=") for a in sys.argv[4:])}
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
cap = cv2.VideoCapture(str(d["path"]))
af = D.AnchorFill()
af.reset(json.load(open("../settings.json"))["anchor_offset"])
af.load_state()
af.debug = True
for k, v in opts.items():
    setattr(af, k, v)
st = {}
orig_guard = af.occlusion_guard


def guard_spy(tex, L, alpha):
    st["pre"] = alpha.copy()
    r = orig_guard(tex, L, alpha)
    st["post"] = r.copy()
    return r


af.occlusion_guard = guard_spy
rows = []
n = 0
y0, y1, x0, x1 = 85, 175, 70, 230
while n <= max(frames):
    ok, f = cap.read()
    if not ok:
        break
    st.clear()
    af.dbg = None
    if d["ok"][n]:
        af.process(f, d["pts"][n].astype(np.float64))
    if n in frames and af.dbg is not None:
        g = af.dbg
        gray = lambda m: cv2.cvtColor((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        comp = np.clip(g["crop"] * (1 - g["alpha"][..., None]) + g["tex"] * g["alpha"][..., None], 0, 255).astype(np.uint8)
        pre = st.get("pre", g["alpha"])
        tiles = [g["crop"], gray(g["mouth"]), gray(pre), gray(g["alpha"]), comp]
        t = np.hstack([x[y0:y1, x0:x1] for x in tiles])
        t = cv2.resize(t, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, f"{n}  crop | mouth | alpha before guard | final alpha | composite   seam {af.seam_s:.1f}", (4, 16), 0, 0.5, (0, 255, 255), 1)
        rows.append(t)
    n += 1
cv2.imwrite(out, np.vstack(rows))
