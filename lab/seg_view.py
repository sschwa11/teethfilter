import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "..")
import dental as D

key = sys.argv[1]
frames = [int(x) for x in sys.argv[2].split(",")]
out = sys.argv[3]
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
cap = cv2.VideoCapture(str(d["path"]))
af = D.AnchorFill()
af.reset(json.load(open("../settings.json"))["anchor_offset"])
af.load_state()
af.debug = True
orig_seg = af.seg
orig_clean = af.clean_mouth
st = {}


def seg_spy(crop):
    m = orig_seg(crop)
    st["raw"] = m.copy()
    st["crop0"] = crop.copy()
    return m


def clean_spy(mouth, cp):
    m = orig_clean(mouth, cp)
    st["clean"] = m.copy()
    band = np.zeros((D.CH, D.CW), np.uint8)
    cv2.fillPoly(band, [np.round(cp[D.INNER_POLY]).astype(np.int32)], 1)
    st["poly"] = band
    return m


af.seg = seg_spy
af.clean_mouth = clean_spy
rows = []
n = 0
y0, y1, x0, x1 = 80, 200, 60, 260
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
        c0 = st["crop0"].copy()
        cs, _ = cv2.findContours(st["poly"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(c0, cs, -1, (0, 255, 255), 1)
        rawm = cv2.cvtColor((np.clip(st["raw"], 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        clean = cv2.cvtColor((np.clip(st["clean"], 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        comp = np.clip(g["crop"] * (1 - g["alpha"][..., None]) + g["tex"] * g["alpha"][..., None], 0, 255).astype(np.uint8)
        t = np.hstack([c0[y0:y1, x0:x1], rawm[y0:y1, x0:x1], clean[y0:y1, x0:x1], comp[y0:y1, x0:x1]])
        t = cv2.resize(t, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        n_comp = cv2.connectedComponentsWithStats((st["raw"] * cv2.GaussianBlur(cv2.dilate(st["poly"], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))).astype(np.float32), (0, 0), 2.0) > 0.5).astype(np.uint8), connectivity=8)[0] - 1
        cv2.putText(t, f"{n}  components {n_comp}  raw>0.5 {int((st['raw'] > 0.5).sum())}  clean>0.5 {int((st['clean'] > 0.5).sum())}", (4, 16), 0, 0.5, (0, 255, 255), 1)
        rows.append(t)
    n += 1
cv2.imwrite(out, np.vstack(rows))
