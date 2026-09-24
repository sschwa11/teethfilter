import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "..")
import dental as D


def gap_series(key, out_paths):
    d = dict(np.load(f"cache/{key}.npz", allow_pickle=True))
    cap_raw = cv2.VideoCapture(str(d["path"]))
    caps = [cv2.VideoCapture(p) for p in out_paths]
    af = D.AnchorFill()
    af.debug = True
    af.upper = False
    af.lower = False
    slot = af.slot > 0.5
    series = [[] for _ in out_paths]
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
            for s in series:
                s.append(np.nan)
            n += 1
            continue
        a, mouth = af.last[0], g["mouth"]
        vis = slot & (mouth > 0.7)
        donor = af.donor_mask & (mouth > 0.7)
        Lr = cv2.cvtColor(g["crop"], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        if vis.sum() < 40 or donor.sum() < 30:
            for s in series:
                s.append(np.nan)
            n += 1
            continue
        tl = float(np.percentile(Lr[donor], 60))
        for s, o in zip(series, outs):
            c = cv2.warpAffine(o, a.astype(np.float32), (D.CW, D.CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            L = cv2.cvtColor(c, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
            s.append(float((L[vis] < 0.6 * tl).mean()))
        n += 1
    return [np.array(s) for s in series]


def blinks(s, hi=0.3, lo=0.15):
    count = 0
    events = []
    for t in range(1, len(s) - 3):
        if np.isnan(s[t]) or np.isnan(s[t - 1]):
            continue
        if s[t] > hi and s[t - 1] < lo:
            after = s[t + 1:t + 3]
            if np.any((after < lo) & ~np.isnan(after)):
                count += 1
                events.append(t)
    return count, events


if __name__ == "__main__":
    key = sys.argv[1]
    paths = sys.argv[2:]
    ser = gap_series(key, paths)
    for p, s in zip(paths, ser):
        c, ev = blinks(s)
        print(json.dumps({"video": p.split("/")[-1], "frames": int(np.isfinite(s).sum()), "mean_gap": round(float(np.nanmean(s)), 3), "gap>0.3": round(float(np.nanmean(s > 0.3)), 3), "blinks": c, "at": ev[:15]}), flush=True)
