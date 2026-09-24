import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "..")
import dental as D


def run(key, setup=None, strip_out=None):
    d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
    cap = cv2.VideoCapture(str(d["path"]))
    af = D.AnchorFill()
    af.debug = True
    if setup:
        setup(af)
    rec = []
    prev = None
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        af.dbg = None
        if d["ok"][n]:
            af.process(f, d["pts"][n].astype(np.float64))
        g = af.dbg
        if g is None:
            rec.append(None)
            prev = None
            n += 1
            continue
        alpha, mouth = g["alpha"], g["mouth"]
        comp = g["crop"].astype(np.float32) * (1 - alpha[..., None]) + g["tex"] * alpha[..., None]
        L = cv2.cvtColor(np.clip(comp, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        Lr = cv2.cvtColor(g["crop"], cv2.COLOR_BGR2GRAY).astype(np.float32)
        r = {"n": n, "area": float(alpha.sum()), "mouth": float(mouth[90:175, 60:260].sum()), "dx": float(g["delta"][0])}
        if prev is not None:
            region = (np.maximum(alpha, prev["alpha"]) > 0.05)
            r["d_alpha"] = float(np.abs(alpha - prev["alpha"])[region].sum())
            r["d_mouth"] = float(np.abs(mouth - prev["mouth"])[region].sum())
            if region.sum() > 50:
                r["d_comp"] = float(np.abs(L - prev["L"])[region].mean())
                r["d_raw"] = float(np.abs(Lr - prev["Lr"])[region].mean())
        prev = {"alpha": alpha, "mouth": mouth, "L": L, "Lr": Lr}
        rec.append(r)
        n += 1
    rs = [r for r in rec if r is not None and "d_alpha" in r]
    pops = 0
    for r, p in zip(rs[1:], rs[:-1]):
        big = max(r["area"], p["area"])
        if big > 150 and abs(r["area"] - p["area"]) > 0.4 * big and abs(r["mouth"] - p["mouth"]) < 0.15 * max(r["mouth"], p["mouth"], 1):
            pops += 1
    da = np.array([r["d_alpha"] for r in rs])
    dm = np.array([r["d_mouth"] for r in rs])
    dc = np.array([r["d_comp"] for r in rs if "d_comp" in r])
    draw = np.array([r["d_raw"] for r in rs if "d_comp" in r])
    dx = np.array([r["dx"] for r in rec if r is not None])
    out = {"frames": len(rs), "pops": pops, "alpha_flicker": round(float(da.sum() / max(dm.sum(), 1)), 3),
           "comp_vs_raw": round(float(dc.mean() / max(draw.mean(), 1e-6)), 3), "dx_jitter": round(float(np.mean(np.abs(np.diff(dx)))), 3)}
    return out, rec


if __name__ == "__main__":
    keys = sys.argv[1:] or ["2026-09-24_06-32-55_raw", "2026-09-24_06-33-17_raw"]
    for k in keys:
        out, _ = run(k)
        print(k, json.dumps(out), flush=True)
