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
    rows = np.arange(D.CH)[:, None]
    slot = af.slot > 0.5
    tooth_rows = (rows >= af.body_top[None, :] + 4) & (rows <= af.bottom_full[None, :] - 4)
    res = [[] for _ in out_paths]
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
        val = None
        if g is not None and af.locks >= 3:
            a = af.last[0]
            cp = D.to_crop(a, d["pts"][n].astype(np.float64))
            poly = np.zeros((D.CH, D.CW), np.uint8)
            cv2.fillPoly(poly, [np.round(cp[D.INNER_POLY]).astype(np.int32)], 1)
            poly = cv2.erode(poly, np.ones((3, 3), np.uint8)) > 0
            zone = slot & tooth_rows & poly
            Lr = cv2.cvtColor(g["crop"], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
            donor = af.donor_mask & poly
            if zone.sum() >= 60 and donor.sum() >= 30:
                ref = float(np.percentile(Lr[donor], 85))
                val = (a, zone, ref)
        if val is None:
            for r in res:
                r.append(np.nan)
        else:
            a, zone, ref = val
            for r, o in zip(res, outs):
                c = cv2.warpAffine(o, a.astype(np.float32), (D.CW, D.CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                L = cv2.cvtColor(c, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
                r.append(float((L[zone] >= 0.6 * ref).mean()))
        n += 1
    return [np.array(r) for r in res]


def dips(s, drop=0.25):
    ev = []
    for t in range(1, len(s) - 2):
        a, b = s[t - 1], s[t]
        c = np.nanmax(s[t + 1:t + 3]) if np.isfinite(s[t + 1:t + 3]).any() else np.nan
        if np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and a - b > drop and c - b > drop:
            ev.append(t)
    return ev


if __name__ == "__main__":
    key = sys.argv[1]
    paths = sys.argv[2:]
    out = series(key, paths)
    np.save(f"out/upcover_{key}.npy", np.stack(out))
    for p, s in zip(paths, out):
        fin = np.isfinite(s)
        pair = fin[1:] & fin[:-1]
        jit = np.abs(np.diff(s))[pair]
        ev = dips(s)
        print(json.dumps({"video": os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p), "frames": int(fin.sum()), "covered": round(float(s[fin].mean()), 3),
                          "jitter": round(float(jit.mean()), 3), "jumps>0.25": int((jit > 0.25).sum()), "dips": len(ev), "at": ev[:40]}), flush=True)
