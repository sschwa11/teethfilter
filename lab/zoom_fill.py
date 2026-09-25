import sys, cv2, numpy as np
sys.path.insert(0, "..")
import dental as D
key = sys.argv[1]; frames = [int(x) for x in sys.argv[2].split(",")]; out = sys.argv[3]
d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
cap = cv2.VideoCapture(str(d["path"]))
af = D.AnchorFill(); af.debug = True
import json, os
if os.environ.get("LIVE"):
    af.reset(json.load(open("../settings.json"))["anchor_offset"]); af.load_state()
for k, v in (a.split("=") for a in sys.argv[4:]):
    setattr(af, k, eval(v))
rows = []; n = 0
y0, y1, x0, x1 = (95, 165, 100, 230) if not os.environ.get("LOW") else (120, 180, 110, 220)
while n <= max(frames):
    ok, f = cap.read()
    if not ok: break
    af.dbg = None
    if d["ok"][n]:
        af.process(f, d["pts"][n].astype(np.float64))
    if n in frames and af.dbg is not None:
        g = af.dbg
        comp = np.clip(g["crop"] * (1 - g["alpha"][..., None]) + g["tex"] * g["alpha"][..., None], 0, 255).astype(np.uint8)
        tiles = [g["crop"][y0:y1, x0:x1], comp[y0:y1, x0:x1], np.clip(g["tex"][y0:y1, x0:x1], 0, 255).astype(np.uint8)]
        if af.st_tex is not None:
            r0, r1, c0, c1 = af.roi
            st = np.zeros_like(g["crop"], np.float32); st[r0:r1, c0:c1] = af.st_tex * af.st_gain
            tiles.append(np.clip(st[y0:y1, x0:x1], 0, 255).astype(np.uint8))
        t = cv2.resize(np.hstack(tiles), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, f"{n} gain {np.round(af.st_gain, 2) if af.st_gain is not None else None}", (4, 16), 0, 0.5, (0, 255, 255), 1)
        cv2.line(t, (int((af.x_mid - x0) * 3), 0), (int((af.x_mid - x0) * 3), 8), (0, 0, 255), 2)
        rows.append(t)
    n += 1
cv2.imwrite(out, np.vstack(rows))
