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
            xm = int(round(af.x_mid))
            band_rows = np.flatnonzero((mouth[:, xm - 10] > 0.7) & (mouth[:, xm + 12] > 0.7) & (np.arange(D.CH) <= af.bottom_full[xm] - 3))
            if ref >= 60 and band_rows.size >= 4:
                rec = (a, band_rows, ref, xm)
        if rec is None:
            for r in res:
                r.append(np.nan)
            info.append(None)
        else:
            a, band_rows, ref, xm = rec
            for r, o in zip(res, outs):
                c = cv2.warpAffine(o, a.astype(np.float32), (D.CW, D.CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                L = cv2.cvtColor(c, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
                widths = []
                for y in band_rows:
                    seg = L[y, xm - 10:xm + 14] >= 0.6 * ref
                    if seg[:4].any() and seg[-4:].any():
                        idx = np.flatnonzero(seg)
                        gaps = np.diff(idx) - 1
                        widths.append(int(gaps.max()) if gaps.size else 0)
                r.append(float(np.median(widths)) if len(widths) >= 3 else np.nan)
            info.append(len(band_rows))
        n += 1
    return [np.array(r) for r in res], info


def events(s, hi=3, lo=1):
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
    np.save(f"out/seam_{key}.npy", np.stack(out))
    for p, s in zip(paths, out):
        ev = events(s)
        fin = np.isfinite(s)
        pair = fin[1:] & fin[:-1]
        jit = np.abs(np.diff(s))[pair]
        print(json.dumps({"video": os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p), "frames": int(fin.sum()), "seam_px": round(float(s[fin].mean()), 2),
                          "seam>=3px": round(float((s[fin] >= 3).mean()), 3), "seam>=5px": round(float((s[fin] >= 5).mean()), 3), "jumps>=3": int((jit >= 3).sum()), "blinks": len(ev), "at": ev[:40]}), flush=True)
