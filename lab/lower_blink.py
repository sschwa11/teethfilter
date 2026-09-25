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
        if g is None:
            for r in res:
                r.append(np.nan)
            n += 1
            continue
        a, mouth = af.last[0], g["mouth"]
        inm = mouth > 0.7
        lb = np.where(inm.any(0), D.CH - 1 - np.argmax(inm[::-1], 0), -1)
        band = inm & (rows > af.bottom_full[None, :] + 2) & (rows < lb[None, :] - 1) & (rows >= lb[None, :] - 22)
        ref_zone = band & ((xs > xm - 60) & (xs < xm - 24))[None, :]
        Lr = cv2.cvtColor(g["crop"], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        ref = float(np.percentile(Lr[ref_zone], 85)) if ref_zone.sum() >= 60 else 0.0
        refb = ref_zone & (Lr >= 0.6 * ref)
        top_t = int(np.median(np.nonzero(refb.any(1))[0][:3])) if refb.any() else D.CH
        zone = band & ((xs > xm - 24) & (xs < xm + 40))[None, :] & (Lr < 0.5 * ref) & (rows >= top_t)
        if zone.sum() < 40 or ref_zone.sum() < 60:
            for r in res:
                r.append(np.nan)
            n += 1
            continue
        if ref < 60:
            for r in res:
                r.append(np.nan)
            n += 1
            continue
        for r, o in zip(res, outs):
            c = cv2.warpAffine(o, a.astype(np.float32), (D.CW, D.CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            L = cv2.cvtColor(c, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
            r.append(float((L[zone] < 0.5 * ref).mean()))
        n += 1
    return [np.array(r) for r in res]


def events(s, hi=0.25, lo=0.1):
    ev = []
    for t in range(1, len(s) - 3):
        if not np.isfinite(s[t]) or not np.isfinite(s[t - 1]):
            continue
        if s[t] > hi and s[t - 1] < lo:
            after = s[t + 1:t + 4]
            if np.any((after < lo) & np.isfinite(after)):
                ev.append(t)
    return ev


if __name__ == "__main__":
    key = sys.argv[1]
    paths = sys.argv[2:]
    out = series(key, paths)
    np.save(f"out/lowblink_{key}.npy", np.stack(out))
    for p, s in zip(paths, out):
        ev = events(s)
        fin = np.isfinite(s)
        pair = fin[1:] & fin[:-1]
        jit = np.abs(np.diff(s))[pair]
        print(json.dumps({"video": os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p), "frames": int(fin.sum()), "gap_left": round(float(s[fin].mean()), 3),
                          "gap>0.3": round(float((s[fin] > 0.3).mean()), 3), "jitter": round(float(jit.mean()), 3), "jumps>0.3": int((jit > 0.3).sum()), "blinks": len(ev), "at": ev[:30]}), flush=True)
