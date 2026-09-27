import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "..")
import dental as D


def series(key, out_paths):
    d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
    cap_raw = cv2.VideoCapture(str(d["path"]))
    caps = [cv2.VideoCapture(p) for p in out_paths]
    af = D.AnchorFill()
    af.reset(json.load(open("../settings.json"))["anchor_offset"])
    af.debug = True
    af.upper = False
    af.lower = False
    xm = af.x_mid
    xs = np.arange(D.CW)
    rows = np.arange(D.CH)[:, None]
    up_cols = (xs >= xm - 90) & (xs < xm - 66)
    lo_cols = (xs > xm + 70) & (xs <= xm + 100)
    res = {z: [[] for _ in out_paths] for z in ("upper", "lower")}
    lip = [[] for _ in out_paths]
    n = 0
    while True:
        ok, raw = cap_raw.read()
        outs = [c.read()[1] for c in caps]
        if not ok or any(o is None for o in outs):
            break
        af.dbg = None
        if d["ok"][n]:
            af.process(raw, d["pts"][n].astype(np.float64))
        g = af.dbg
        if g is None or af.locks < 3:
            n += 1
            continue
        a, mouth = af.last[0], g["mouth"]
        Lr = cv2.cvtColor(g["crop"], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        inm = mouth > 0.7
        donor = af.donor_mask & inm
        ref = float(np.percentile(Lr[donor], 85)) if donor.sum() >= 30 else 0.0
        lb = np.where(inm.any(0), D.CH - 1 - np.argmax(inm[::-1], 0), -1)
        up_rows = (rows >= 88) & (rows <= af.bottom_full[None, :] - 2)
        lo_rows = (rows > af.bottom_full[None, :] + 3) & (rows < lb[None, :] - 1) & (rows >= lb[None, :] - 22)
        zones = {"upper": inm & up_rows & up_cols[None, :] & (Lr < 0.5 * ref),
                 "lower": inm & lo_rows & lo_cols[None, :] & (Lr < 0.5 * ref)}
        lipz = (mouth < 0.2) & ((up_cols | lo_cols)[None, :]) & (rows >= 80) & (rows <= 190)
        outs_c = []
        for o in outs:
            c = cv2.warpAffine(o, a.astype(np.float32), (D.CW, D.CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            outs_c.append(cv2.cvtColor(c, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32))
        for zn, zm in zones.items():
            ok_z = ref >= 60 and zm.sum() >= 30
            for r, L in zip(res[zn], outs_c):
                r.append(float((L[zm] < 0.5 * ref).mean()) if ok_z else np.nan)
        for r, L in zip(lip, outs_c):
            r.append(float((np.abs(L[lipz] - Lr[lipz]) > 25).mean()) if lipz.sum() > 50 else np.nan)
        n += 1
    return res, lip


if __name__ == "__main__":
    key = sys.argv[1]
    paths = sys.argv[2:]
    res, lip = series(key, paths)
    for i, p in enumerate(paths):
        out = {"video": os.path.basename(os.path.dirname(p))}
        for zn in ("upper", "lower"):
            s = np.array(res[zn][i])
            fin = np.isfinite(s)
            out[zn] = f"frames {int(fin.sum())} gap_left {s[fin].mean():.3f} gap>0.3 {(s[fin] > 0.3).mean():.3f}" if fin.any() else "none"
        lp = np.array(lip[i])
        out["lip_painted"] = round(float(np.nanmean(lp)), 4)
        print(json.dumps(out), flush=True)
