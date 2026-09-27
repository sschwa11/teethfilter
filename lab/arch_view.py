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
rows = []
n = 0
y0, y1, x0, x1 = 70, 215, 30, 300
while n <= max(frames):
    ok, f = cap.read()
    if not ok:
        break
    af.dbg = None
    if d["ok"][n]:
        af.process(f, d["pts"][n].astype(np.float64))
    if n in frames and af.dbg is not None:
        g = af.dbg
        crop = g["crop"].copy()
        comp = np.clip(g["crop"] * (1 - g["alpha"][..., None]) + g["tex"] * g["alpha"][..., None], 0, 255).astype(np.uint8)
        xm = af.x_mid
        for img in (crop, comp):
            for dx, col in ((0, (0, 255, 255)), (-66, (0, 0, 255)), (66, (0, 0, 255)), (-88, (255, 0, 255)), (88, (255, 0, 255))):
                cv2.line(img, (int(round(xm + dx)), y0), (int(round(xm + dx)), y0 + 12), col, 1)
            if af.lax is not None:
                for dx, col in ((0, (0, 255, 0)), (af.l_span, (0, 255, 0))):
                    cv2.line(img, (int(round(af.lax + dx)), y1 - 12), (int(round(af.lax + dx)), y1), col, 1)
        t = np.vstack([crop[y0:y1, x0:x1], comp[y0:y1, x0:x1]])
        t = cv2.resize(t, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.putText(t, f"{n}", (4, 16), 0, 0.5, (0, 255, 255), 1)
        rows.append(t)
    n += 1
cv2.imwrite(out, np.hstack(rows) if len(rows) <= 2 else np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows) - 1, 2)] + ([np.hstack([rows[-1], np.zeros_like(rows[-1])])] if len(rows) % 2 else [])))
print("written", len(rows))
