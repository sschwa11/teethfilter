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
orig = af.lower_fill
state = {}


def spy(*a, **k):
    r = orig(*a, **k)
    state["lf"] = r
    return r


af.lower_fill = spy
xs = np.arange(D.CW)
rows = np.arange(D.CH)[:, None]
recs = []
n = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    state.clear()
    af.dbg = None
    if d["ok"][n]:
        af.process(f, d["pts"][n].astype(np.float64))
    g = af.dbg
    if g is not None and "lf" in state:
        xm = af.x_mid
        mouth = g["mouth"]
        inm = mouth > 0.7
        lb = np.where(inm.any(0), D.CH - 1 - np.argmax(inm[::-1], 0), -1)
        band = inm & (rows > af.bottom_full[None, :] + 2) & (rows < lb[None, :] - 1) & (rows >= lb[None, :] - 22)
        ref_zone = band & ((xs > xm - 60) & (xs < xm - 24))[None, :]
        Lr = cv2.cvtColor(g["crop"], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        ref = float(np.percentile(Lr[ref_zone], 85)) if ref_zone.sum() >= 60 else 0.0
        zone = band & ((xs > xm - 24) & (xs < xm + 40))[None, :] & (Lr < 0.5 * ref)
        if zone.sum() >= 40 and ref >= 60:
            comp = g["crop"] * (1 - g["alpha"][..., None]) + g["tex"] * g["alpha"][..., None]
            Lc = cv2.cvtColor(np.clip(comp, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
            gap = float((Lc[zone] < 0.5 * ref).mean())
            al = g["alpha_l"]
            cover = float((al[zone] > 0.5).mean())
            raw_l = state["lf"][1] if state["lf"] is not None else None
            rawcov = float((raw_l[zone] > 0.5).mean()) if raw_l is not None else 0.0
            zx = np.nonzero(zone.any(0))[0]
            recs.append({"n": n, "gap": round(gap, 3), "none": state["lf"] is None, "cover": round(cover, 3), "rawcov": round(rawcov, 3),
                         "lax": None if af.lax is None else round(af.lax, 1), "zx": [int(zx.min()), int(zx.max())], "l_hits": af.l_hits})
    n += 1
json.dump(recs, open(f"out/why_{key}.json", "w"))
g = np.array([r["gap"] for r in recs])
bad = [r for r in recs if r["gap"] > 0.3]
print(key, "frames", len(recs), "gap_left", round(g.mean(), 3), "bad", len(bad))
if bad:
    none = sum(r["none"] for r in bad)
    held = sum((not r["none"]) and r["rawcov"] < 0.3 for r in bad)
    part = sum((not r["none"]) and r["rawcov"] >= 0.3 for r in bad)
    print("  bad: lower_fill None", none, "| returned but covers <30% of gap", held, "| covers >=30% but gap still dark", part)
