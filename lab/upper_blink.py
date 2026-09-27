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
    res = [[] for _ in out_paths]
    info = []
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
        rec = None
        if g is not None and af.locks >= 3:
            a, mouth = af.last[0], g["mouth"]
            Lr = cv2.cvtColor(g["crop"], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
            donor = af.donor_mask & (mouth > 0.7)
            ref = float(np.percentile(Lr[donor], 85)) if donor.sum() >= 30 else 0.0
            zone = slot & (mouth > 0.7) & (rows <= af.bottom_full[None, :] + 1) & (Lr < 0.5 * ref)
            if ref >= 60 and zone.sum() >= 40:
                rec = (a, zone, ref)
        if rec is None:
            for r in res:
                r.append(np.nan)
            info.append(None)
        else:
            a, zone, ref = rec
            for r, o in zip(res, outs):
                c = cv2.warpAffine(o, a.astype(np.float32), (D.CW, D.CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                L = cv2.cvtColor(c, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
                r.append(float((L[zone] < 0.5 * ref).mean()))
            info.append(int(zone.sum()))
        n += 1
    return [np.array(r) for r in res], info


def events(s, hi=0.35, lo=0.15):
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
    out, info = series(key, paths)
    np.save(f"out/upblink_{key}.npy", np.stack(out))
    for p, s in zip(paths, out):
        ev = events(s)
        fin = np.isfinite(s)
        pair = fin[1:] & fin[:-1]
        jit = np.abs(np.diff(s))[pair]
        print(json.dumps({"video": os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p), "frames": int(fin.sum()), "gap_left": round(float(s[fin].mean()), 3),
                          "gap>0.3": round(float((s[fin] > 0.3).mean()), 3), "jitter": round(float(jit.mean()), 3), "jumps>0.3": int((jit > 0.3).sum()), "blinks": len(ev), "at": ev[:40]}), flush=True)
