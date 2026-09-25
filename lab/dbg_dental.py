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
while n <= max(frames):
    ok, f = cap.read()
    if not ok: break
    af.dbg = None
    if d["ok"][n]:
        af.process(f, d["pts"][n].astype(np.float64))
    if n in frames and af.dbg is not None:
        g = af.dbg
        crop = g["crop"].copy()
        cs, _ = cv2.findContours((g["mouth"] > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(crop, cs, -1, (0, 255, 0), 1)
        comp = np.clip(g["crop"] * (1 - g["alpha"][..., None]) + g["tex"] * g["alpha"][..., None], 0, 255).astype(np.uint8)
        al = cv2.cvtColor((g["alpha"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        t = cv2.resize(np.hstack([crop[80:190, 40:280], comp[80:190, 40:280], al[80:190, 40:280]]), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, f"{key[11:19]} {n} d{np.round(g['delta'], 1)} miss {g['miss']} locks {g['locks']} gate {g['gate']:.2f}", (4, 16), 0, 0.5, (0, 255, 255), 1)
        rows.append(t)
    n += 1
cv2.imwrite(out, np.vstack(rows))
