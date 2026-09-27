import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "..")
import dental as D
from parsers import Parser

key = sys.argv[1]
frames = [int(x) for x in sys.argv[2].split(",")]
out = sys.argv[3]
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
cap = cv2.VideoCapture(str(d["path"]))
af = D.AnchorFill()
af.reset(json.load(open("../settings.json"))["anchor_offset"])
af.load_state()
af.debug = True
parsers = [Parser("segformer"), Parser("bisenet")]
st = {}
orig_seg = af.seg


def seg_spy(crop):
    m = orig_seg(crop)
    st["student"] = m.copy()
    return m


af.seg = seg_spy
rows = []
y0, y1, x0, x1 = 80, 200, 60, 260
gray = lambda m: cv2.cvtColor((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
n = 0
while n <= max(frames):
    ok, f = cap.read()
    if not ok:
        break
    st.clear()
    af.dbg = None
    pts = d["pts"][n].astype(np.float64)
    if d["ok"][n]:
        af.process(f, pts)
    if n in frames and af.dbg is not None:
        a0 = af.df.affine(af.df.pose(pts))
        crop0 = cv2.warpAffine(f, a0.astype(np.float32), (D.CW, D.CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        tiles = [crop0, gray(st["student"])]
        for p in parsers:
            probs, m, _ = p.probs(f, pts)
            dm = p.to_dental(probs, m, a0)
            tiles.append(gray(dm["mouth"]))
        t = np.hstack([x[y0:y1, x0:x1] for x in tiles])
        t = cv2.resize(t, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, f"{n}  crop | student | SegFormer mouth | BiSeNet mouth", (4, 16), 0, 0.5, (0, 255, 255), 1)
        rows.append(t)
    n += 1
cv2.imwrite(out, np.vstack(rows))
print("written")
